import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2 as cv
import torch

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