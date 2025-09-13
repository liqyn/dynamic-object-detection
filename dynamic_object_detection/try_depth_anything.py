import os 
import sys 
import cv2
import torch

project_path = '/home/jrached/cv_project_code'
sys.path.append(os.path.join(project_path, 'Depth-Anything-V2'))
# from depth_anything_v2.dpt import DepthAnythingV2
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb' # or 'vits', 'vitb', 'vitg'
dataset = 'vkitti'
max_depth = 80
# max_depth = 10

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
# model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(os.path.join(project_path, f'Depth-Anything-V2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth'), map_location='cpu'))
# model.load_state_dict(torch.load(os.path.join(project_path, f'Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth'), map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread(os.path.join(project_path, "project/data/KITTI_tracking/unzipped/data_tracking_image_3/training/image_03/0000/000050.png"))
# depth = model.infer_image(cv2.resize(raw_img, (640, 640)), input_size=1600) # HxW raw depth map in numpy
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
cv2.imshow("raw_image", raw_img)
print(depth.shape)
print(depth)
depth_vis = depth.copy()
depth_vis[depth_vis > max_depth] = max_depth   # clip far values
depth_vis = (depth_vis / max_depth * 255).astype("uint8")
cv2.imshow("predicted_depth", depth_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()