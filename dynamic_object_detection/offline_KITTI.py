import sys
import os
import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dynamic_object_detection.params_KITTI import Params
from dynamic_object_detection.raft_wrapper import RaftWrapper
from dynamic_object_detection.viz import OpticalFlowVisualizer
from dynamic_object_detection.flow_KITTI import GeometricOpticalFlow
from dynamic_object_detection.tracker import DynamicObjectTracker
from dynamic_object_detection.dod_util import copy_params_file, preprocess_depth, compute_relative_poses
import numpy as np
from tqdm import tqdm
import torch
import gc
import time
import pickle
import warnings
import cv2

project_path = '/home/jrached/cv_project_code'
sys.path.append(os.path.join(project_path, 'Depth-Anything-V2'))
# from depth_anything_v2.dpt import DepthAnythingV2
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

# import our KITTI helpers
from dynamic_object_detection.process_KITTI import (
    load_calib, load_oxts, oxts_to_poses, load_image_paths, load_stereo_pair, disparity_to_depth, stereo_matcher, cam_to_imu, get_depth, CameraParams 
)

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--params', '-p', type=str, default='', help='Path to the parameters file')
    parser.add_argument('--base', type=str, required=True, help='Base path to KITTI dataset')
    parser.add_argument('--seq', type=str, required=True, help='KITTI sequence number (e.g., 0003)')

    args = parser.parse_args()

    params = Params.from_yaml(args.params)

    copy_params_file(params, args)

    print('loading RAFT...')
    raft = RaftWrapper(params.raft_params, params.device)
    print(f'raft loaded with model {params.raft_params.model}')

    print('loading DepthAnything...')
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vits' # or 'vits', 'vitb', 'vitg'
    dataset = 'vkitti'
    max_depth = 80
    depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_model.load_state_dict(torch.load(os.path.join(project_path, f'Depth-Anything-V2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth'), map_location='cpu'))
    depth_model = depth_model.to(params.device).eval()

    print('loading KITTI data...')
    left_dir = f"{args.base}/data_tracking_image_2/training/image_02/{args.seq}"
    right_dir = f"{args.base}/data_tracking_image_3/training/image_03/{args.seq}"
    calib_file = f"{args.base}/calib/training/calib/{args.seq}.txt"
    oxts_file = f"{args.base}/data_tracking_oxts/training/oxts/{args.seq}.txt"

    calib = load_calib(calib_file)
    oxts = load_oxts(oxts_file)
    cam_poses = np.stack(oxts_to_poses(oxts), axis=0)

    left_paths = load_image_paths(left_dir)
    right_paths = load_image_paths(right_dir)

    matcher = stereo_matcher()

    # derive times (KITTI has no explicit timestamps in tracking set just frame indices)
    times = np.arange(len(left_paths)) / params.original_fps
    times = times[::params.skip_frames]
    N_frames = len(times)

    # Load all images + depths
    imgs = []
    depth_imgs = []
    for i in range(0, len(left_paths), params.skip_frames):
        left = cv2.imread(left_paths[i])[2:-5, 1:-1, :]
        right = cv2.imread(right_paths[i])[2:-5, 1:-1, :]
        # disp = matcher.compute(
        #     cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
        #     cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        # ).astype(np.float32) / 16.0
        # depth = disparity_to_depth(disp, calib['P2'], calib['P3'])
        depth = get_depth(depth_model, left) 

        imgs.append(left)
        depth_imgs.append(preprocess_depth(depth, params.depth_data_params))

    imgs = np.stack(imgs, axis=0)             # (N, H, W, 3)
    depth_imgs = np.stack(depth_imgs, axis=0) # (N, H, W)
    T_1_0 = compute_relative_poses(torch.tensor(cam_poses, dtype=torch.float32, device=params.device)) # (N-1, 4, 4)

    # Apply cam to imu transforms to relative poses
    T_cam_to_imu = torch.from_numpy(cam_to_imu(calib)).to(device=params.device, dtype=torch.float32)
    T_imu_to_cam = torch.inverse(T_cam_to_imu)

    T_1_0 = T_imu_to_cam.unsqueeze(0) @ T_1_0 @ T_cam_to_imu.unsqueeze(0)
    
    print(f'Loaded {N_frames} frames from KITTI sequence {args.seq}')
    # ----------------------------------------------------

    effective_fps = params.original_fps / params.skip_frames
    runtimes = []

    H, W, _ = right.shape 
    depth_params = CameraParams(calib['P2'][:, :3], H, W)
    if params.viz_params.viz_video:
        viz = OpticalFlowVisualizer(params.viz_params, f'{params.output}.mp4', effective_fps)
    gof_flow = GeometricOpticalFlow(depth_params, device=params.device)  # pass K from calib
    tracker = DynamicObjectTracker(params.tracking_params, depth_params, effective_fps)

    for index in tqdm(range(0, N_frames - 1, params.batch_size)):
        torch.cuda.empty_cache()
        gc.collect()

        batch_end = min(index + params.batch_size, N_frames - 1)

        batch_imgs_np = imgs[index:batch_end+1]
        batch_imgs = torch.tensor(batch_imgs_np, dtype=torch.float32, device=params.device)
        batch_img0s = batch_imgs[:-1]
        batch_img1s = batch_imgs[1:]
        batch_depth_imgs_np = depth_imgs[index:batch_end+1]
        batch_depth_imgs = torch.tensor(batch_depth_imgs_np, dtype=torch.float32, device=params.device)
        batch_T_1_0 = T_1_0[index:batch_end]

        start_time = time.time()

        raft_flows = raft.run_raft_batch(batch_img0s, batch_img1s)
        residual, coords_3d, raft_coords_3d_1, geom_flows = gof_flow.compute_flow(
            raft_flows, batch_depth_imgs, batch_T_1_0, use_3d=params.use_3d
        )

        raft_flows = raft_flows.cpu().numpy()
        if geom_flows is not None: geom_flows = geom_flows.cpu().numpy()
        residual = residual.cpu().numpy()
        coords_3d = coords_3d.cpu().numpy()
        raft_coords_3d_1 = raft_coords_3d_1.cpu().numpy()

        dynamic_masks, orig_dynamic_masks = tracker.run_tracker(
            residual, 
            batch_imgs_np[:-1],
            batch_depth_imgs_np[:-1],
            coords_3d,
            raft_coords_3d_1,
            times=times[index:batch_end],
            cam_poses=cam_poses[index:batch_end],
            draw_objects=params.viz_params.viz_video and params.viz_params.viz_dynamic_object_masks,
        )

        runtimes.append(time.time() - start_time)

        if params.viz_params.viz_video:
            viz.write_batch_frames(
                batch_imgs_np[:-1],
                batch_depth_imgs_np[:-1],
                dynamic_masks,
                orig_dynamic_masks,
                raft_flows,
                geom_flows,
                residual,
            )

    if params.viz_params.viz_video:
        viz.end()

    out = {
        'objects': tracker.return_objects(),
        'times': times[:-1],
        'poses': cam_poses,
        'runtime': {
            'avg_batch_time': np.mean(runtimes),
            'avg_frame_time': np.mean(runtimes) / params.batch_size,
            'total_time': np.sum(runtimes),
        },
        'camera_info': {
            'K': calib['P2'],  # intrinsics from calib
            'H': imgs.shape[1],
            'W': imgs.shape[2],
        }
    }

    with open(f'{params.output}.pkl', 'wb') as fout:
        pickle.dump(out, fout)
        print(f'output and info saved to {params.output}.pkl')
