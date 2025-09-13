import os
import numpy as np
import cv2
# import pykitti
from pyproj import Proj

import sys 
import torch

class CameraParams: 
    def __init__(self, K, H, W):
        self.K = K
        self.height = H 
        self.width = W 

def get_depth(model, left_img):
    max_depth = 80
    depth = model.infer_image(left_img)
    depth[depth > max_depth] = max_depth
    return depth 


# --------------------------
# CONFIGURATION
# --------------------------
def load_calib(calib_file):
    data = {}
    with open(calib_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            key = parts[0]

            # Case 1: "P0: ..." style
            if ':' in line:
                key, value = line.split(':', 1)
                vals = [float(x) for x in value.strip().split()]
                data[key] = np.array(vals).reshape(3, 4)

            # Case 2: "R_rect ..." or "Tr_velo_cam ..." style
            else:
                vals = [float(x) for x in parts[1:]]
                if key.startswith("R"):      # 3x3 rectification
                    data[key] = np.array(vals).reshape(3, 3)
                else:                        # 3x4 transforms
                    data[key] = np.array(vals).reshape(3, 4)
    return data

def load_oxts(oxts_file):
    oxts = []
    with open(oxts_file, 'r') as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            oxts.append(vals)
    return np.array(oxts)

def oxts_to_poses(oxts):
    # UTM projection for KITTI (zone 32N, WGS84 ellipsoid)
    proj_utm = Proj(proj="utm", zone=32, ellps="WGS84")

    lat0, lon0, alt0 = oxts[0][:3]
    x0, y0 = proj_utm(lon0, lat0)

    poses = []
    for row in oxts:
        lat, lon, alt, roll, pitch, yaw = row[:6]
        x, y = proj_utm(lon, lat)
        t = np.array([x - x0, y - y0, alt - alt0])  # relative translation in meters

        # Rotation matrix (yaw-pitch-roll, ZYX order)
        Rx = np.array([[1,0,0],
                       [0,np.cos(roll),-np.sin(roll)],
                       [0,np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],
                       [0,1,0],
                       [-np.sin(pitch),0,np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],
                       [np.sin(yaw), np.cos(yaw),0],
                       [0,0,1]])
        R = Rz @ Ry @ Rx

        pose = np.eye(4)
        pose[:3,:3] = R
        pose[:3, 3] = t
        poses.append(pose)
    return poses


def load_image_paths(image_dir):
    return sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])

def load_stereo_pair(left_path, right_path):
    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    return left, right

def load_labels(label_file):
    """Parse KITTI tracking labels into a dict of frame_idx -> list of dicts"""
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if not vals:
                continue
            frame = int(vals[0])
            track_id = int(vals[1])
            obj_type = vals[2]
            truncated = float(vals[3])
            occluded = int(vals[4])
            alpha = float(vals[5])
            bbox = [float(v) for v in vals[6:10]]
            dims = [float(v) for v in vals[10:13]]  # h, w, l
            loc  = [float(v) for v in vals[13:16]]  # x, y, z in camera coords
            ry   = float(vals[16])

            entry = {
                "track_id": track_id,
                "type": obj_type,
                "truncated": truncated,
                "occluded": occluded,
                "alpha": alpha,
                "bbox": bbox,
                "dims": dims,
                "loc": loc,
                "ry": ry,
            }
            if frame not in labels:
                labels[frame] = []
            labels[frame].append(entry)
    return labels


def stereo_matcher():
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8*3*5**2,
        P2=32*3*5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

def disparity_to_depth(disparity, P2, P3):
    f = P2[0,0]
    B = abs(P2[0,3] - P3[0,3]) / f
    mask = disparity > 0
    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[mask] = f * B / disparity[mask]
    return depth

def object_cam_to_enu(obj, cam_pose_enu, calib):
    """
    Convert KITTI label (camera coords) to ENU global frame.

    obj: dict from label file with keys 'loc' and 'ry'
    cam_pose_enu: 4x4 matrix, camera pose in ENU (from OXTS pipeline)
    calib: dict from load_calib()

    Returns:
        obj_pose_enu: 4x4 SE(3) matrix, object pose in ENU
    """
    # --- Object location in camera coords ---
    x, y, z = obj['loc']
    ry = obj['ry']

    # Homogeneous point
    p_cam = np.array([x, y, z, 1.0])

    # --- Extrinsics ---
    # Rectification
    R_rect = np.eye(4)
    R_rect[:3,:3] = calib['R_rect']

    # Velodyne->camera
    T_velo_cam = np.eye(4)
    T_velo_cam[:3,:] = calib['Tr_velo_cam']

    # IMU->Velodyne
    T_imu_velo = np.eye(4)
    T_imu_velo[:3,:] = calib['Tr_imu_velo']

    # Camera->IMU
    T_cam_to_imu = np.linalg.inv(R_rect @ T_velo_cam) @ T_imu_velo

    # Camera->ENU = ENU_pose_of_imu * (cam->imu)
    T_cam_to_enu = cam_pose_enu @ T_cam_to_imu

    # --- Object orientation in camera frame ---
    R_obj_cam = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [ 0,          1, 0         ],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    T_obj_cam = np.eye(4)
    T_obj_cam[:3,:3] = R_obj_cam
    T_obj_cam[:3, 3] = [x, y, z]

    # --- Transform into ENU ---
    T_obj_enu = T_cam_to_enu @ T_obj_cam
    return T_obj_enu

def cam_to_imu(calib):
        # --- Extrinsics ---
    # Rectification
    R_rect = np.eye(4)
    R_rect[:3,:3] = calib['R_rect']

    # Velodyne->camera
    T_velo_cam = np.eye(4)
    T_velo_cam[:3,:] = calib['Tr_velo_cam']

    # IMU->Velodyne
    T_imu_velo = np.eye(4)
    T_imu_velo[:3,:] = calib['Tr_imu_velo']

    # Camera->IMU
    return np.linalg.inv(R_rect @ T_velo_cam @ T_imu_velo)

def estimate_vel_from_labels(labels, track_id, poses, dt=0.1):
    """
    Estimate velocity of an object in ENU frame using finite differences.

    labels: dict[frame_idx] -> list of objects (from load_labels)
    track_id: int, object track ID
    poses: list of ego camera poses in ENU (from oxts_to_poses)
    dt: timestep between frames (KITTI is 10 Hz -> dt=0.1s)

    Returns:
        dict[frame_idx] -> velocity magnitude (m/s)
    """
    vel_dict = {}
    prev_pos = None
    prev_frame = None

    for frame_idx in sorted(labels.keys()):
        objs = [o for o in labels[frame_idx] if o["track_id"] == track_id]
        if not objs:
            continue

        # Convert this object's pose to ENU
        obj = objs[0]
        T_obj_enu = object_cam_to_enu(obj, poses[frame_idx], calib=None)  # pass calib if needed
        pos = T_obj_enu[:3, 3]

        if prev_pos is not None:
            dt_actual = (frame_idx - prev_frame) * dt
            vel = np.linalg.norm(pos - prev_pos) / dt_actual
            vel_dict[frame_idx] = vel

        prev_pos = pos
        prev_frame = frame_idx

    return vel_dict



if __name__ == '__main__':
    base = '/home/jrached/cv_project_code/project/data/KITTI_tracking/unzipped'    
    seq = "0003"
    left_dir = f"{base}/data_tracking_image_2/training/image_02/{seq}"
    right_dir = f"{base}/data_tracking_image_3/training/image_03/{seq}"
    calib_file = f"{base}/calib/training/calib/{seq}.txt"
    oxts_file = f"{base}/data_tracking_oxts/training/oxts/{seq}.txt"
    label_file = f"{base}/data_tracking_label_2/training/label_02/{seq}.txt"

    # Load everything
    num_frames = 2000
    calib = load_calib(calib_file)
    oxts = load_oxts(oxts_file)
    poses = oxts_to_poses(oxts)
    left_paths, right_paths = load_image_paths(left_dir), load_image_paths(right_dir)
    matcher = stereo_matcher()
    labels = load_labels(label_file)

    # Example processing
    try: 
        for i in range(num_frames):
            left, right = load_stereo_pair(left_paths[i], right_paths[i])
            disp = matcher.compute(left, right).astype(np.float32) / 16.0
            depth = disparity_to_depth(disp, calib['P2'], calib['P3'])

            if i in labels:
                for obj in labels[i]:
                    if obj["type"] not in ["DontCare", "Misc"]:
                        # --- 2D bounding box overlay ---
                        x1, y1, x2, y2 = map(int, obj["bbox"])
                        cv2.rectangle(left, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(left, f"{obj['type']} {obj['track_id']}",
                                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                        # --- Convert GT to ENU ---
                        pose_enu = poses[i]   # camera pose in ENU
                        T_obj_enu = object_cam_to_enu(obj, pose_enu, calib)

                        # Print (or save) ENU position
                        print(f"Frame {i}, ID {obj['track_id']} {obj['type']}")
                        print("  ENU translation:", T_obj_enu[:3,3])

            # --- Visualization ---
            disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow("Left with GT", left)
            cv2.imshow("Disparity", disp_vis)

            if cv2.waitKey(0) == 27:
                break
    except IndexError: 
        print("Index out of range")

    print("End of sequence")
    cv2.destroyAllWindows()
