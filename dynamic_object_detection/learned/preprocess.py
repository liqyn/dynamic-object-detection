import argparse
from dynamic_object_detection.params import Params
from dynamic_object_detection.raft_wrapper import RaftWrapper
from dynamic_object_detection.viz import OpticalFlowVisualizer
from dynamic_object_detection.flow import GeometricOpticalFlow
from dynamic_object_detection.dod_util import copy_params_file, preprocess_depth, compute_relative_poses
from dynamic_object_detection.learned.dataloader import GTExtractor
import numpy as np
from tqdm import tqdm
import os
import torch
import gc
import time
import pickle
import cv2

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--params', '-p', type=str, default='', help='Path to the parameters file')

    args = parser.parse_args()

    params = Params.from_yaml(args.params)

    parent_dir = os.path.dirname(params.output) if os.path.dirname(params.output) else '.'
    os.makedirs(parent_dir, exist_ok=True)

    copy_params_file(params, args)

    print('loading raft...')
    raft = RaftWrapper(params.raft_params, params.device)
    print(f'raft loaded with model {params.raft_params.model}')

    print('loading data...')
    cam_pose_data = params.load_camera_pose_data()
    gt_cam_pose_data = params.gt_params.load_gt_camera_pose_data()
    img_data = params.load_img_data()
    depth_data = params.load_depth_data()
    model_params = params.model_params
    print('pose, img, depth data loaded')

    times = img_data.times[::params.skip_frames]
    N_frames = len(times)
    imgs = np.stack([img_data.img(t) for t in times], axis=0) # (N, H, W, 3)
    depth_imgs = np.stack([preprocess_depth(depth_data.img(t), params.depth_data_params) for t in times], axis=0) # (N, H, W)
    cam_poses = np.stack([cam_pose_data.pose(t) for t in times], axis=0)
    gt_cam_poses = np.stack([gt_cam_pose_data.pose(t) for t in times], axis=0)
    T_0_1 = compute_relative_poses(torch.tensor(cam_poses, dtype=torch.float32, device=params.device)) # (N-1, 4, 4)

    print(f'running algorithm on {N_frames} frames from t={times[0]} to t={times[-1]}')

    effective_fps = params.original_fps / params.skip_frames

    if params.viz_params.viz_video:
        viz = OpticalFlowVisualizer(params.viz_params, f'{params.output}.mp4', effective_fps)

    gof_flow = GeometricOpticalFlow(depth_data.camera_params, device=params.device)
    
    gt_extractor = GTExtractor(
        params=params.gt_params,
        times=times,
        images=imgs,
        cam_poses=gt_cam_poses,
        camera_params=img_data.camera_params,
        device=params.device
    )

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

        raft_flows_2d = raft.run_raft_batch(batch_img1s, batch_img0s) # (B H W 2)

        _, residual_2d, geometric_flows_2d = gof_flow.compute_residual_2d_flow(raft_flows_2d, 
                                                                  batch_depth_imgs[1:], batch_T_0_1, norm=False) # (B H W 2), (B H W 2)
        
        raft_coords_3d_1 = gof_flow.raft_unproject(raft_flows_2d, batch_depth_imgs[:-1])
        N = len(raft_coords_3d_1)

        coords_3d, geometric_coords_3d_1 = gof_flow.unproject_and_transform(batch_depth_imgs[1:], batch_T_0_1)
        raft_flows_3d = (raft_coords_3d_1 - coords_3d).view(N, gof_flow.H, gof_flow.W, 3) # (B H W 3) 
        geometric_flows_3d = (geometric_coords_3d_1 - coords_3d).view(N, gof_flow.H, gof_flow.W, 3) # (B H W 3) 
        residual_flow_3d = raft_flows_3d - geometric_flows_3d # (B H W 3)

        tensor_list = []
        
        if model_params.use_raft_flows_3d: tensor_list.append(raft_flows_3d)
        if model_params.use_geometric_flows_3d: tensor_list.append(geometric_flows_3d)
        if model_params.use_residual_flow_3d: tensor_list.append(residual_flow_3d)

        # 2D Flows
        if model_params.use_raft_flows_2d: tensor_list.append(raft_flows_2d)
        if model_params.use_geometric_flows_2d: tensor_list.append(geometric_flows_2d)
        if model_params.use_residual_2d: tensor_list.append(residual_2d)

        if model_params.use_depth_data: 
            tensor_list.append(batch_depth_imgs[1:].unsqueeze(-1))
            if model_params.use_last_frame:
                tensor_list.append(batch_depth_imgs[:-1].unsqueeze(-1))
        if model_params.use_image_data: 
            tensor_list.append(batch_img1s.to(dtype=torch.float32) / 255.0)
            if model_params.use_last_frame:
                tensor_list.append(batch_img0s.to(dtype=torch.float32) / 255.0)

        combined_flow_batch = torch.cat(tensor_list, dim=-1) # (B H W, CC)
        
        masks, moving_robots_list, bboxes_list = gt_extractor.get_dynamic_objects(index, batch_end) # (B H W)
        

        viz_imgs = batch_imgs_np[1:]

        for i in range(batch_end - index):
            bboxes = bboxes_list[i]
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(viz_imgs[i], (x1, y1), (x2, y2), (0, 0, 255), 2)

        if params.viz_params.viz_video:
            # print('writing video frames...')
            viz.write_batch_frames(
                image=viz_imgs,
                depth=batch_depth_imgs_np[1:],
                dynamic_mask=masks,
                orig_dynamic_masks=masks,
                raft_flow=raft_flows_2d.cpu().numpy(),
                geom_flow=geometric_flows_2d.cpu().numpy(),
                residual=np.linalg.norm(residual_2d.cpu().numpy(), axis=-1),
            )
            
        for i in range(batch_end - index):
            input_save_name = os.path.join(params.gt_params.inputs_output_dir, f'{index + i}.pt')
            torch.save(combined_flow_batch[i].to(dtype=torch.float32).cpu(), input_save_name)
            mask_save_name = os.path.join(params.gt_params.gt_masks_output_dir, f'{index + i}.pt')
            torch.save(torch.from_numpy(masks[i].astype(np.uint8)).cpu(), mask_save_name)
            
    if params.viz_params.viz_video: viz.end()
