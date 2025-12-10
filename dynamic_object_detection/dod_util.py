from dynamic_object_detection.learned.dataloader import GTExtractor
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2 as cv
import torch
import json

def global_nearest_neighbor(data1: list, data2: list, cost_fn: callable, max_cost: float=None):
    """
    Associates data1 with data2 using the global nearest neighbor algorithm.

    Args:
        data1 (list): List of first data items
        data2 (list): List of second data items
        cost_fn (callable(item1, item2)): Function to compute the cost of associating two objects
        max_cost (float): Maximum cost to consider association

    Returns:
        a dictionary d such that d[i] = j means that data2[i] is associated with data1[j]
    """
    len1 = len(data1)
    len2 = len(data2)
    if len1 == 0 or len2 == 0: return {}

    # Augment cost to add option for no associations
    hungarian_cost = np.zeros((len1, len2))
    M = 1e9

    max_used_score = -float('inf')

    for i in range(len1):
        for j in range(len2):
            score = cost_fn(data1[i], data2[j]) # Hungarian is trying to associate low scores, no negation needed
            
            if max_cost is not None and score > max_cost:
                score = M
            else:
                max_used_score = max(max_used_score, score)
            hungarian_cost[i,j] = score

    if max_used_score == -float('inf'):
        max_used_score = 1

    expanded_cost = np.full((2 * len1, 2 * len2), max_used_score)
    expanded_cost[:len1, :len2] = hungarian_cost
    hungarian_cost = expanded_cost

    row_ind, col_ind = linear_sum_assignment(hungarian_cost)

    assignment = {}

    for idx1, idx2 in zip(row_ind, col_ind):
        if idx1 < len1 and idx2 < len2:
            assignment[idx2] = idx1

    return assignment

def copy_params_file(params, args):
    params_copy_path = f'{params.output}.yaml'
    with open(args.params, 'r') as src_file, open(params_copy_path, 'w') as dest_file:
        dest_file.write(src_file.read())
    print(f'saved params file to {params_copy_path}')

def preprocess_depth(depth, depth_params):
    depth = depth.astype(np.float32)

    if depth_params.bilateral_smooth_depth is not None:
        d, sigmaColor, sigmaSpace = depth_params.bilateral_smooth_depth
        depth = cv.bilateralFilter(depth, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    depth = depth / depth_params.depth_scale

    if depth_params.max_depth is not None: depth[depth > depth_params.max_depth] = 0
    return depth

def compute_relative_poses(poses):
    pose0_inv = torch.stack([torch.linalg.inv(pose) for pose in poses[:-1]], dim=0)                         # (N, 4, 4)
    pose1 = poses[1:]

    # T_0_1 = T_w_0^-1 @ T_w_1: transform from frame 1 to frame 0  
    T_0_1 = pose0_inv @ pose1                                                                               # (N-1, 4, 4)
    
    return T_0_1

def remove_nan_points(points):
    return points[~np.isnan(points).any(axis=1)]

def transform_points(T_w_frame, points):
    """
    Transform points from camera frame to worlod frame
    T_w_frame: (4, 4)
    points: (N, 3)
    """
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    return (T_w_frame @ points_h.T)[:3].T


def dice_coeff(input: np.ndarray, target: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Compute the Dice coefficient between two 2D numpy arrays.
    
    Args:
        input: np.ndarray of shape (H, W), binary or float mask in [0,1]
        target: np.ndarray of shape (H, W), same dtype/shape as input
        epsilon: small number to avoid division by zero
    
    Returns:
        dice: float, Dice coefficient
    """
    assert input.shape == target.shape, "Input and target must have the same shape"

    inter = 2 * np.sum(input * target)
    sets_sum = np.sum(input) + np.sum(target)

    # Handle empty masks
    if sets_sum == 0:
        sets_sum = inter  # dice = 1 if both empty

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice

class GTEvalTracker:
    def __init__(self, params, times, imgs, rgb_camera_params, gt_tracker, device):
        self.params = params
        self.times = times
        gt_cam_pose_data = params.gt_params.load_gt_camera_pose_data()
        self.gt_cam_poses = np.stack([gt_cam_pose_data.pose(t) for t in times], axis=0)

        # TODO: make this also input? or get rid of circular import with tracker
        self.gt_extractor = GTExtractor(
            params=params.gt_params,
            times=times,
            images=imgs,
            cam_poses=self.gt_cam_poses,
            camera_params=rgb_camera_params,
            device=device
        )

        self.gt_tracker = gt_tracker

        self.gt_eval = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'Dice coefficient': [],
        }
        
    def eval_batch(self, index, res_pred_masks, batch_imgs_np, batch_depth_imgs_np, coords_3d, raft_coords_3d_1):
        
        for frame in range(len(res_pred_masks)):
            
            frame_mask, _ = self.gt_tracker.run_tracker(
                res_pred_masks[frame:frame+1].copy(),
                batch_imgs_np[frame+1:frame+2].copy(),
                batch_depth_imgs_np[frame+1:frame+2].copy(),
                coords_3d[frame:frame+1].copy(),
                raft_coords_3d_1[frame:frame+1].copy(),
                times=self.times[index + frame:index + frame + 1].copy(),
                cam_poses=self.gt_cam_poses[index + frame:index + frame + 1].copy(),
                draw_objects=False,
            )

            # Dice coefficient
            masks, moving_robots_list, _ = self.gt_extractor.get_dynamic_objects(index + frame, index + frame + 1) # (1 H W)
            gt_mask = masks[0]

            self.gt_eval['Dice coefficient'].append(dice_coeff(frame_mask[0], gt_mask))


            # recall and precision
            gt_robot_poses = self.gt_extractor.get_gt_robots(index + frame)

            seen_tracked_object_ids = set()
            true_positives = 0

            for rid, pose in gt_robot_poses.items():
                if rid not in moving_robots_list[0]: continue
                if pose[0, 3] > self.params.depth_data_params.max_depth: continue
                for tid, to in self.gt_tracker.tracked_objects.items():
                    # if tid in seen_tracked_object_ids: continue
                    dist = np.linalg.norm(to.cur_point - pose[:3, 3])
                    if dist < self.params.gt_params.gt_dist_threshold:
                        seen_tracked_object_ids.add(tid)
                        true_positives += 1
                        # break

            false_positives = len(self.gt_tracker.tracked_objects) - true_positives
            false_negatives = len(moving_robots_list[0]) - true_positives

            self.gt_eval['true_positives'] += true_positives
            self.gt_eval['false_positives'] += false_positives
            self.gt_eval['false_negatives'] += false_negatives

    def save_eval_results(self, output_path):
        self.gt_eval['Dice coefficient'] = float(np.mean(self.gt_eval['Dice coefficient']))
        with open(output_path, 'w') as f:
            json.dump(self.gt_eval, f, indent=4)
            print(f'gt evaluation saved to {output_path}_gt_eval.json')