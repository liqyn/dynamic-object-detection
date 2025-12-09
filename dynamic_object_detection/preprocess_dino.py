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
import torch.nn.functional as F

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
    
    print('loading dino')
    dino_patch_size = 16
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(params.device)

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
    
    #Dino cannot process too many at the same time
    batch_size = 1
    for index in tqdm(range(1, N_frames, batch_size)):
        torch.cuda.empty_cache()
        gc.collect()

        batch_end = min(index + batch_size, N_frames)
        # print(f'processing frames {index} to {batch_end}')

        batch_imgs_np = imgs[index:batch_end]
        batch_imgs = torch.tensor(batch_imgs_np, dtype=torch.float32, device=params.device) # (B H W 3)
        batch_depth_imgs_np = depth_imgs[index:batch_end]
        batch_depth_imgs = torch.tensor(batch_depth_imgs_np, dtype=torch.float32, device=params.device) # (B H W)
        batch_T_0_1 = T_0_1[index:batch_end] # (B 4 4)

        start_time = time.time()
         
        batch_first_imgs = batch_imgs.permute(0, 3, 1, 2)
        dino_features = dino(batch_first_imgs)
        dino_features = dino.get_intermediate_layers(batch_first_imgs, len(dino.blocks))
        dino_features = dino_features[-1]
       
        dino_patch_tokens = dino_features[:, 1:, :]
        _, N_tokens, D = dino_patch_tokens.shape

        _, H_orig, W_orig, _ = batch_imgs.shape

        H_patch = H_orig // dino_patch_size
        W_patch = W_orig // dino_patch_size

        dino_reshaped = dino_patch_tokens.permute(0, 2, 1).reshape(dino_patch_tokens.shape[0], D, H_patch, W_patch)
        scaled_dino_features = F.interpolate(
            dino_reshaped,
            size=(H_orig, W_orig),
            mode='bilinear',
            align_corners=False
        )
        scaled_dino_features_channels_last = scaled_dino_features.permute(0, 2, 3, 1)
         
        batch_size = batch_end - index
        for batch_idx in range(batch_size):
            frame_number = index + batch_idx
            single_frame_tensor = scaled_dino_features_channels_last[batch_idx].cpu()
            filename = f"{params.output}{frame_number:05d}dino.pt"
            torch.save(single_frame_tensor, filename)
    print("done outputting all of the dino frames")
