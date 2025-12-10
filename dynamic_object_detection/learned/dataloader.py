import os
import math
import numpy as np
import cv2
import torch
import torch.utils.data as torch_data
from scipy.spatial.transform import Rotation as R

from segment_anything import sam_model_registry, SamPredictor

from robotdatapy.data.pose_data import PoseData
from robotdatapy.data.img_data import ImgData
from robotdatapy.transform import T_FLURDF

from dynamic_object_detection.params import GTParams


class GTExtractor:
    def __init__(
        self,
        params: GTParams,
        times: np.ndarray,
        images: np.ndarray,
        cam_poses: np.ndarray,
        camera_params,
        device='cuda',
    ):
        super().__init__()

        self.vel_dt = params.gt_vel_dt
        self.speed_threshold = params.speed_threshold
        self.cache_dir = params.cache_dir
        self.camera_params = camera_params

        sam = sam_model_registry[params.sam_model_type](checkpoint=params.sam_checkpoint)
        sam.to(device=device)
        sam.eval()

        # Create a global predictor
        self.sam_predictor = SamPredictor(sam)

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.obstacle_robots_config = params.obstacle_robots_config
        self.robot_ids = list(self.obstacle_robots_config.keys())

        # Load pose streams for each obstacle robot from pose bag
        self.obstacle_pose_data = {}
        for rid, cfg in self.obstacle_robots_config.items():
            pose_topic = cfg["pose_topic"]
            self.obstacle_pose_data[rid] = PoseData.from_bag(
                params.gt_pose_data.params_dict['path'],
                topic=pose_topic,
                time_tol=params.gt_pose_data.params_dict['time_tol'],
            )

        # Build a list of valid camera indices where all poses are available
        self.cam_times = times
        self.images = images
        self.cam_poses = cam_poses

    def get_dynamic_objects(self, idx_start=0, idx_end=None):
        if idx_end is None:
            idx_end = len(self.cam_times)

        masks = []
        moving_robots_list = []
        bboxes_list = []

        for idx in range(idx_start, idx_end):
            data = self.__getitem__(idx)
            masks.append(data["mask"])  # (H, W)
            moving_robots_list.append(data["moving_robots"])
            bboxes_list.append(data["bboxes"])

        masks = np.stack(masks, axis=0)  # (N, H, W)
        return masks, moving_robots_list, bboxes_list

    def __len__(self):
        return len(self.cam_times)

    def __getitem__(self, idx):
        t = self.cam_times[idx]

        time_str = f"{t:.6f}"

        img_bgr = self.images[idx]
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        H, W, _ = img_rgb.shape

        # Check cache
        mask = None
        moving_robots = []
        bboxes = []
        cache_path = None
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f"mask_{time_str}.npz")
            if os.path.isfile(cache_path):
                arr = np.load(cache_path)
                mask = arr["mask"].astype(np.uint8)
                moving_robots = list(arr["moving_robots"])
                bboxes = list(map(tuple, arr["bboxes"]))

        if mask is None:
            T_world_cam = self.cam_poses[idx]
            mask, moving_robots, bboxes = self._build_dynamic_mask(
                img_rgb, t, T_world_cam, H, W
            )

            if cache_path is not None:
                np.savez_compressed(
                    cache_path,
                    mask=mask.astype(np.uint8),
                    moving_robots=np.array(moving_robots, dtype="U16"),
                    bboxes=np.array(bboxes, dtype=np.int32),
                )

        # mask = mask.astype(np.float32)

        return {
            # "image": img_t,
            "mask": mask,
            "moving_robots": moving_robots,
            "bboxes": bboxes,
        }
    
    def get_gt_robots(self, idx):
        t = self.cam_times[idx]

        T_world_ego = self.cam_poses[idx]

        robot_poses = {}
        for rid in self.robot_ids:
            T_world_robot = self.obstacle_pose_data[rid].pose(t)
            T_world_robot[2, 3] += self.obstacle_robots_config[rid].get("z_offset", 0.0)
            robot_poses[rid] = np.linalg.inv(T_world_ego) @ T_world_robot # T_ego_robot

        return robot_poses


    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------

    def _estimate_speed(self, rid, t):
        """Finite difference speed estimate for robot rid at time t."""
        pose_data = self.obstacle_pose_data[rid]

        idx1 = pose_data.idx(t - self.vel_dt, force_single=True)
        idx2 = pose_data.idx(t + self.vel_dt, force_single=True)

        if idx1 == idx2: return None

        t1 = pose_data.times[idx1]
        t2 = pose_data.times[idx2]
        pose1 = pose_data.pose(t1)
        pose2 = pose_data.pose(t2)

        return np.linalg.norm(pose2[:3, 3] - pose1[:3, 3]) / (t2 - t1)

    def _project_bbox(self, T_world_cam, T_world_robot, dims, img_shape):
        """
        Project a 3D box with dims=(L,W,H) at T_world_robot into the image.

        Returns xyxy=(x_min, y_min, x_max, y_max) or None.
        """
        H, W = img_shape[:2]
        fx, fy, cx, cy = (
            self.camera_params.fx,
            self.camera_params.fy,
            self.camera_params.cx,
            self.camera_params.cy,
        )

        L, W_r, H_r = dims
        hx, hy, hz = L / 2.0, W_r / 2.0, H_r / 2.0

        corners_robot = np.array(
            [
                [hx, hy, hz],
                [hx, hy, -hz],
                [hx, -hy, hz],
                [hx, -hy, -hz],
                [-hx, hy, hz],
                [-hx, hy, -hz],
                [-hx, -hy, hz],
                [-hx, -hy, -hz],
            ]
        )

        T_world_robot = np.asarray(T_world_robot)
        T_world_cam = np.asarray(T_world_cam)
        T_cam_world = np.linalg.inv(T_world_cam)
        T_cam_robot = T_cam_world @ T_world_robot

        R_cr = T_cam_robot[:3, :3]
        t_cr = T_cam_robot[:3, 3]

        pts_cam = (R_cr @ corners_robot.T + t_cr[:, None]).T  # (8,3)
        z = pts_cam[:, 2]
        valid = z > 0.05
        if not np.any(valid):
            return None

        pts_cam = pts_cam[valid]
        x = pts_cam[:, 0]
        y = pts_cam[:, 1]
        z = pts_cam[:, 2]

        u = fx * x / z + cx
        v = fy * y / z + cy

        u_min, v_min = np.min(u), np.min(v)
        u_max, v_max = np.max(u), np.max(v)

        u_min = int(max(0, math.floor(u_min)))
        v_min = int(max(0, math.floor(v_min)))
        u_max = int(min(W - 1, math.ceil(u_max)))
        v_max = int(min(H - 1, math.ceil(v_max)))

        if u_min >= u_max or v_min >= v_max:
            return None

        return (u_min, v_min, u_max, v_max)

    def _build_dynamic_mask(self, img_rgb, t, T_world_cam, H, W):
        """
        Determine which robots are dynamic at time t and build mask via SAM.
        """
        mask = np.zeros((H, W), dtype=np.uint8)
        moving_robots = []
        bboxes = []

        for rid in self.robot_ids:
            speed = self._estimate_speed(rid, t)
            if speed < self.speed_threshold:
                continue  # static

            dims = self.obstacle_robots_config[rid].get("dims", None)
            z_offset = self.obstacle_robots_config[rid].get("z_offset", 0.0)
            if dims is None:
                continue

            T_world_robot = self.obstacle_pose_data[rid].pose(t)
            T_world_robot[2, 3] += z_offset
            bbox = self._project_bbox(T_world_cam, T_world_robot, dims, img_shape=img_rgb.shape)
            if bbox is None:
                continue

            # enlarge bbox slightly
            x1, y1, x2, y2 = bbox
            enlarge = 0.1
            w_box = x2 - x1
            h_box = y2 - y1
            dx = int(enlarge * w_box)
            dy = int(enlarge * h_box)

            x1 = max(0, x1 - dx)
            y1 = max(0, y1 - dy)
            x2 = min(W - 1, x2 + dx)
            y2 = min(H - 1, y2 + dy)
            bbox = (x1, y1, x2, y2)

            # Run SAM with box prompt
            sam_mask = self.sam_predict(img_rgb, bbox)  # HxW bool or uint8
            sam_mask = sam_mask.astype(bool)

            mask[sam_mask] = 1
            moving_robots.append(rid)
            bboxes.append(bbox)

        return mask, moving_robots, bboxes
        

    def sam_predict(self, image_rgb: np.ndarray, box_xyxy):
        """
        Args:
            image_rgb: HxWx3 uint8 array in RGB order.
            box_xyxy: (x1, y1, x2, y2), in pixel coordinates.

        Returns:
            mask: HxW uint8 array, values {0,1}, where 1 indicates the segmented robot.
        """
        # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        self.sam_predictor.set_image(image_rgb)

        box = np.array(box_xyxy, dtype=np.float32)[None, :]  # shape (1,4)

        masks, scores, logits = self.sam_predictor.predict(
            box=box,
            point_coords=None,
            point_labels=None,
            multimask_output=False
        )

        mask = masks[0].astype(np.uint8)  # (H, W) bool

        return mask
