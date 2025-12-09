import argparse
from dynamic_object_detection.params import Params
from dynamic_object_detection.raft_wrapper import RaftWrapper
from dynamic_object_detection.viz import OpticalFlowVisualizer
from dynamic_object_detection.flow import GeometricOpticalFlow
from dynamic_object_detection.tracker import DynamicObjectTracker
from dynamic_object_detection.dod_util import copy_params_file, preprocess_depth, compute_relative_poses
import numpy as np
from tqdm import tqdm
import os
import torch
import gc
import time
import pickle

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

    gof_flow = GeometricOpticalFlow(depth_data.camera_params, device=params.device)
    tracker = DynamicObjectTracker(params.tracking_params, depth_data.camera_params, effective_fps)

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

        gflow_2d, residual_2d = gof_flow.compute_residual_2d_flow_not_normalized(raft_flows_2d, batch_depth_imgs[1:], batch_T_0_1) # (B H W 2), (B H W 2)
        
        raft_coords_3d_1 = gof_flow.raft_unproject(raft_flows_2d, batch_depth_imgs[:-1])
        N = len(raft_coords_3d_1)

        coords_3d, geometric_coords_3d_1 = gof_flow.unproject_and_transform(batch_depth_imgs[1:], batch_T_0_1)
        raft_flows_3d = (raft_coords_3d_1 - coords_3d).view(N, gof_flow.H, gof_flow.W, 3) # (B H W 3) 
        geometric_flows_3d = (geometric_coords_3d_1 - coords_3d).view(N, gof_flow.H, gof_flow.W, 3) # (B H W 3) 
        residual_flow = raft_flows_3d - geometric_flows_3d # (B H W 3)

        batch_first_imgs = batch_imgs.permute(0, 3, 1, 2)

        raft_flows_2d = raft_flows_2d.cpu()
        gflow_2d = gflow_2d.cpu()
        residual_2d = residual_2d.cpu()

        tensor_list = []
        
        tensor_list.append(raft_flows_2d)

        tensor_list.append(gflow_2d)

        tensor_list.append(residual_2d)
        
        tensor_list.append(batch_depth_imgs[1:].cpu().unsqueeze(-1)) #cut start off as that is index - 1
        
        tensor_list.append(batch_img1s.cpu())

        combined_flow_batch = torch.cat(tensor_list, dim=-1)
        
        batch_size = batch_end - index
        for batch_idx in range(batch_size):
            frame_number = index + batch_idx
            single_frame_tensor = combined_flow_batch[batch_idx].cpu()
            filename = f"{params.output}{frame_number:05d}.pt"
            torch.save(single_frame_tensor, filename)
