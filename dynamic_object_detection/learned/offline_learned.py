import argparse
from dynamic_object_detection.params import Params
from dynamic_object_detection.raft_wrapper import RaftWrapper
from dynamic_object_detection.viz import OpticalFlowVisualizer
from dynamic_object_detection.flow import GeometricOpticalFlow
from dynamic_object_detection.tracker import DynamicObjectTracker
from dynamic_object_detection.dod_util import copy_params_file, preprocess_depth, compute_relative_poses, GTEvalTracker
from dynamic_object_detection.learned.dataloader import GTExtractor
import numpy as np
from tqdm import tqdm
import os
import sys
import torch
import gc
import time
import pickle
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--params', '-p', type=str, default='', help='Path to the parameters file')

    args = parser.parse_args()

    params = Params.from_yaml(args.params)


    parent_dir = os.path.dirname(params.output) if os.path.dirname(params.output) else '.'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    copy_params_file(params, args)

    print('loading raft...')
    raft = RaftWrapper(params.raft_params, params.device)
    print(f'raft loaded with model {params.raft_params.model}')

    print('loading data...')
    cam_pose_data = params.load_camera_pose_data()
    img_data = params.load_img_data()
    depth_data = params.load_depth_data()
    model_params = params.model_params
    print('pose, img, depth data loaded')

    # print(len(cam_pose_data.positions))

    times = img_data.times[::params.skip_frames]
    N_frames = len(times)
    imgs = np.stack([img_data.img(t) for t in times], axis=0) # (N, H, W, 3)
    depth_imgs = np.stack([preprocess_depth(depth_data.img(t), params.depth_data_params) for t in times], axis=0) # (N, H, W)
    cam_poses = np.stack([cam_pose_data.pose(t) for t in times], axis=0)
    T_0_1 = compute_relative_poses(torch.tensor(cam_poses, dtype=torch.float32, device=params.device)) # (N-1, 4, 4)

    print(f'running algorithm on {N_frames} frames from t={times[0]} to t={times[-1]}')

    effective_fps = params.original_fps / params.skip_frames

    runtimes = []

    # Load pre-trained UNet
    sys.path.append(params.model_params.unet_source_dir)
    from unet import UNet

    model = UNet(n_channels=model_params.num_channels, n_classes=1, bilinear=False).to(params.device)
    state_dict = torch.load(model_params.checkpoint, map_location=params.device)
    del state_dict['mask_values']
    model.load_state_dict(state_dict)
    model.eval()

    if params.viz_params.viz_video:
        viz = OpticalFlowVisualizer(params.viz_params, f'{params.output}.mp4', effective_fps)
    gof_flow = GeometricOpticalFlow(depth_data.camera_params, device=params.device)
    tracker = DynamicObjectTracker(params.tracking_params, depth_data.camera_params, effective_fps)

    # load gt
    if params.run_gt_eval: 
        gt_tracker = DynamicObjectTracker(params.tracking_params, depth_data.camera_params, effective_fps)
        gt_eval_tracker = GTEvalTracker(params, times, imgs, img_data.camera_params, gt_tracker, params.device)
        

    all_res_pred_masks = []

    for index in tqdm(range(1, N_frames, params.batch_size)):

        torch.cuda.empty_cache()
        gc.collect()

        batch_end = min(index + params.batch_size, N_frames)

        # print(f'processing frames {index} to {batch_end}')

        batch_imgs_np = imgs[index - 1:batch_end]
        batch_imgs = torch.tensor(batch_imgs_np, dtype=torch.float32, device=params.device) # (B+1 H W 3)
        batch_img0s = batch_imgs[:-1] # (B H W 3)
        batch_img1s = batch_imgs[1:] # (B H W 3)
        batch_depth_imgs_np = depth_imgs[index - 1:batch_end]
        batch_depth_imgs = torch.tensor(batch_depth_imgs_np, dtype=torch.float32, device=params.device) # (B+1 H W)
        batch_T_0_1 = T_0_1[index - 1:batch_end - 1] # (B 4 4)

        start_time = time.time()

        raft_flows_2d = raft.run_raft_batch(batch_img1s, batch_img0s) # (B H W 2)

        raft_coords_3d_1 = gof_flow.raft_unproject(raft_flows_2d, batch_depth_imgs[:-1])
        N = len(raft_coords_3d_1)

        tensor_list = []

        coords_3d = None
        
        if model_params.use_raft_flows_3d or model_params.use_geometric_flows_3d or model_params.use_residual_flow_3d:

            coords_3d, geometric_coords_3d_1 = gof_flow.unproject_and_transform(batch_depth_imgs[1:], batch_T_0_1)
            raft_flows_3d = (raft_coords_3d_1 - coords_3d).view(N, gof_flow.H, gof_flow.W, 3) # (B H W 3) 
            geometric_flows_3d = (geometric_coords_3d_1 - coords_3d).view(N, gof_flow.H, gof_flow.W, 3) # (B H W 3) 
            residual_flow_3d = raft_flows_3d - geometric_flows_3d # (B H W 3)
            
            if model_params.use_raft_flows_3d: tensor_list.append(raft_flows_3d)
            if model_params.use_geometric_flows_3d: tensor_list.append(geometric_flows_3d)
            if model_params.use_residual_flow_3d: tensor_list.append(residual_flow_3d)

        # 2D Flows
        if model_params.use_raft_flows_2d or model_params.use_geometric_flows_2d or model_params.use_residual_2d:
            coords_3d, residual_2d, geometric_flows_2d = gof_flow.compute_residual_2d_flow(raft_flows_2d, 
                                                        batch_depth_imgs[1:], batch_T_0_1, norm=False)
            

            if model_params.use_raft_flows_2d: tensor_list.append(raft_flows_2d)

            # TODO: reverse this when training data is fixed
            if model_params.use_residual_2d: tensor_list.append(residual_2d)
            if model_params.use_geometric_flows_2d: tensor_list.append(geometric_flows_2d)

        if model_params.use_depth_data: 
            tensor_list.append(batch_depth_imgs[1:].unsqueeze(-1))
            if model_params.use_last_frame:
                tensor_list.append(batch_depth_imgs[:-1].unsqueeze(-1))
        if model_params.use_image_data: 
            tensor_list.append(batch_img1s.to(dtype=torch.float32) / 255.0)
            if model_params.use_last_frame:
                tensor_list.append(batch_img0s.to(dtype=torch.float32) / 255.0)

        combined_flow_batch = torch.cat(tensor_list, dim=-1).permute(0, 3, 1, 2) # (B CC H W)

        nan_mask = torch.isnan(combined_flow_batch)
        combined_flow_batch[nan_mask] = 0.0

        with torch.no_grad():
            res_pred_masks = model(combined_flow_batch.to(dtype=torch.float32, device=params.device)).squeeze(1) # (B H W)

        res_pred_masks = (torch.sigmoid(res_pred_masks) > 0.5).to(dtype=torch.float32)

        out_nan_mask = torch.any(nan_mask, dim=1) # (B H W)
        res_pred_masks[out_nan_mask] = torch.nan

        res_pred_masks = res_pred_masks.detach().cpu().numpy() # (B H W)

        all_res_pred_masks.append(res_pred_masks)

        if coords_3d is None:
            coords_3d = gof_flow.unproject(batch_depth_imgs[1:]) # (B H W 3)

        coords_3d = coords_3d.cpu().numpy()
        raft_coords_3d_1 = raft_coords_3d_1.cpu().numpy()

        # print('running dynamic object tracker...')
        dynamic_masks, orig_dynamic_masks = tracker.run_tracker(
            res_pred_masks, 
            batch_imgs_np[1:],
            batch_depth_imgs_np[1:],
            coords_3d,
            raft_coords_3d_1,
            times=times[index:batch_end],
            cam_poses=cam_poses[index:batch_end],
            draw_objects=params.viz_params.viz_video and params.viz_params.viz_dynamic_object_masks,
        )

        runtimes.append(time.time() - start_time)

        if params.run_gt_eval:
            gt_eval_tracker.eval_batch(index, res_pred_masks, batch_imgs_np, batch_depth_imgs_np, coords_3d, raft_coords_3d_1)

        if params.viz_params.viz_video:
            # print('writing video frames...')
            viz.write_batch_frames(
                batch_imgs_np[1:],
                batch_depth_imgs_np[1:],
                dynamic_masks,
                orig_dynamic_masks,
                None,
                None,
                None,
            )

    if params.viz_params.viz_video: viz.end()


    all_res_pred_masks = np.concatenate(all_res_pred_masks, axis=0) # (N-1, H, W)

    out = {
        'objects': tracker.return_objects(),
        'times': times[1:], # first frame is not tracked
        'poses': cam_poses, # includes first frame
        'residual_pred_masks': all_res_pred_masks,
        'runtime': {
            'avg_batch_time': np.mean(runtimes),
            'avg_frame_time': np.mean(runtimes) / params.batch_size,
            'total_time': np.sum(runtimes),
        },
        'camera_info': {
            'K': depth_data.camera_params.K,
            'H': depth_data.camera_params.height,
            'W': depth_data.camera_params.width,
        }
    }

    with open(f'{params.output}.pkl', 'wb') as fout:
        pickle.dump(out, fout)
        print(f'output and info saved to {params.output}.pkl')

    if params.run_gt_eval:
        gt_eval_tracker.save_eval_results(f'{params.output}_gt_eval.json')
