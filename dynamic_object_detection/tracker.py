import cv2 as cv
import numpy as np

from dynamic_object_detection.dod_util import global_nearest_neighbor, remove_nan_points
from scipy.ndimage import label

TEXT_OVERLAY = (0, 255, 0)
BBOX_OVERLAY = (0, 0, 255)
OVERLAY = (np.array([255, 0, 0], dtype=np.uint8))

def global_nearest_neighbor_dynamic_objects(tracked_objects: dict, new_objects: list, cost_fn: callable, max_cost: float=None):
    """
    Associates tracked objects with new objects using the global nearest neighbor algorithm.

    Args:
        tracked_objects (dict): Dictionary of tracked objects
        new_objects (list): List of new objects
        cost_fn (callable): Function to compute the cost of associating two objects
        max_cost (float): Maximum cost to consider association

    Returns:
        a dictionary d such that d[i] = j means that new_object with id i is associated with tracked_object with id j
    """
    tracked_objects_list = list(tracked_objects.values())
    assignment = global_nearest_neighbor(tracked_objects_list, new_objects, cost_fn, max_cost)
    id_assignment = {new_objects[i].id : tracked_objects_list[j].id for i, j in assignment.items()}
    return id_assignment


class DynamicObjectTracker:
    def __init__(self, params, depth_camera_params, effective_fps):
        self.params = params
        self.H = depth_camera_params.height
        self.W = depth_camera_params.width
        self.min_residual_threshold = params.min_vel_threshold / effective_fps
        self.residual_threshold_gain = params.vel_threshold_gain / effective_fps

        self.tracked_objects = {}
        self.all_objects = {}
        self._id = 0
        self.cur_frame = 0

    def run_tracker(self, residuals, imgs, depths, coords_3d, raft_coords_3d_1, times, cam_poses, draw_objects=False):
        """
        residuals: (N, H, W)
        imgs: (N, H, W, 3)
        depths: (N, H, W)
        coords_3d: (N, H*W, 3)
        raft_coords_3d_1: (N, H*W, 3)
        cam_poses: (N, 4, 4)
        times: (N,)
        """
        dynamic_mask, orig_dynamic_mask = self.compute_dynamic_mask_batch(residuals, depths)  # (N, H, W)

        for frame in range(len(imgs)):
            
            labeled_mask, num_features = label(dynamic_mask[frame], structure=[[0,1,0],[1,1,1],[0,1,0]])

            new_objects, new_object_cur_points = self.labeled_mask_to_objects(labeled_mask, num_features, coords_3d[frame], raft_coords_3d_1[frame], cam_poses[frame])

            associations = global_nearest_neighbor_dynamic_objects(
                tracked_objects=self.tracked_objects, 
                new_objects=new_objects, 
                cost_fn=DynamicObjectTrack.distance, 
                max_cost=self.params.max_merge_dist,
            )

            # print(f'{len(new_objects)} new objects, {len(self.tracked_objects)} tracked objects, {len(associations)} associated')

            to_remove_ids = [obj.id for obj in self.tracked_objects.values() if obj.id not in associations.values()]
            self.remove_dynamic_objects(to_remove_ids)

            for new_obj, new_obj_cur_points in zip(new_objects, new_object_cur_points):
                if new_obj.id in associations:
                    to_update_id = associations[new_obj.id]
                else:
                    to_update_id = new_obj.id
                    self.tracked_objects[new_obj.id] = new_obj
                    self.all_objects[new_obj.id] = new_obj # track all over time

                self.tracked_objects[to_update_id].update_trajectory(new_obj.cur_point, self.cur_frame) # add to tracked object trajectory
                self.tracked_objects[to_update_id].update(new_obj.mask, new_obj_cur_points) # update mask and current points

            mask = np.zeros((self.H, self.W), dtype=bool)

            if draw_objects:
                imgs[frame][mask] = imgs[frame][mask] * 0.4 + OVERLAY * 0.6
                
                for obj in self.tracked_objects.values():
                    obj_mask = obj.mask.reshape((self.H, self.W)).astype(np.uint8)
                    mask = np.logical_or(mask, obj_mask)

                    coords = np.argwhere(obj_mask > 0)  # coordinates of mask
                    ymin, xmin = coords.min(axis=0)
                    ymax, xmax = coords.max(axis=0)

                    imgs[frame] = cv.rectangle(
                        imgs[frame],
                        (xmin, ymin),
                        (xmax, ymax),
                        color=BBOX_OVERLAY,
                        thickness=2
                    )

                    obj_mask_center = np.mean(coords, axis=0).astype(int)
                    imgs[frame] = cv.putText(imgs[frame], str(obj.id), (obj_mask_center[1], obj_mask_center[0]), cv.FONT_HERSHEY_SIMPLEX, 1.5, TEXT_OVERLAY, 2)

            self.cur_frame += 1

        return dynamic_mask, orig_dynamic_mask
    
    def labeled_mask_to_objects(self, labeled_mask, num_features, coords_3d_frame, raft_coords_3d_1, cam_pose):
        """
        labeled_mask: (H, W)
        coords_3d_frame: (H*W, 3)
        """
        objects = []
        object_cur_points = []
        for i in range(1, num_features + 1):
            mask = (labeled_mask == i).reshape((-1)).astype(bool)
            prev_points = remove_nan_points(raft_coords_3d_1[mask])
            cur_points = remove_nan_points(coords_3d_frame[mask])

            if len(prev_points) < 4 or len(cur_points) < 4: continue
            if self.bad_dynamic_object_check(prev_points): continue

            obj = DynamicObjectTrack(self._id, mask, prev_points)
            self._id += 1
            objects.append(obj)
            object_cur_points.append(cur_points)

        return objects, object_cur_points
    
    def bad_dynamic_object_check(self, points):
        centered_points = points - points.mean(axis=0)
        cov = np.cov(centered_points.T)

        # print(f'cov: {cov}')

        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd, axes])

        if self.params.max_3d_std_dev is not None and np.sqrt(np.max(cov.diagonal())) > self.params.max_3d_std_dev: return True
        if self.params.min_3d_std_dev is not None and np.sqrt(np.max(cov.diagonal())) < self.params.min_3d_std_dev: return True
        

    def remove_dynamic_objects(self, to_remove_ids):
        for id_ in to_remove_ids:
            del self.tracked_objects[id_]

    def compute_dynamic_mask_batch(self, residuals, depths):

        # Pre-processing

        if self.params.gaussian_smoothing:
            smoothed_residuals = np.zeros_like(residuals)
            for i in range(residuals.shape[0]):
                smoothed_residuals[i] = cv.GaussianBlur(residuals[i], (self.params.gaussian_kernel_size, self.params.gaussian_kernel_size), 0)
            residuals = smoothed_residuals

        threshold = self.min_residual_threshold + self.residual_threshold_gain * depths  # (N, H, W)
        mask = ((residuals > threshold) & ~np.isnan(residuals)).astype(np.uint8)  # (N, H, W)

        # Post-processing

        orig_mask = mask.copy()

        for method, kernel_size in self.params.post_processing:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            for i in range(mask.shape[0]):
                if method == 'dilate':
                    mask[i] = cv.dilate(mask[i], kernel, iterations=1)
                elif method == 'erode':
                    mask[i] = cv.erode(mask[i], kernel, iterations=1)
                elif method == 'open':
                    mask[i] = cv.morphologyEx(mask[i], cv.MORPH_OPEN, kernel)
                elif method == 'close':
                    mask[i] = cv.morphologyEx(mask[i], cv.MORPH_CLOSE, kernel)

        return mask, orig_mask
    
    def return_objects(self):
        frame_object_list = [[] for _ in range(self.cur_frame)]
        for obj in self.all_objects.values():
            if len(obj.points_list) >= self.params.min_consecutive_frames:
                for point, frame in zip(obj.points_list, obj.frames):
                    frame_object_list[frame].append({'id': obj.id, 'point': point})
        return frame_object_list

    

class DynamicObjectTrack:
    def __init__(self, id_, mask, points):
        self.id = id_
        self.points_list = []
        self.frames = []
        self.update(mask, points)

    def update(self, mask, points):
        self.mask = mask
        self.cur_point = np.mean(points, axis=0) # centroid
        self.points = points

    def update_trajectory(self, cur_point, frame):
        self.points_list.append(cur_point)
        self.frames.append(frame)

    @classmethod
    def distance(cls, obj1, obj2):
        return np.linalg.norm(obj1.cur_point - obj2.cur_point)