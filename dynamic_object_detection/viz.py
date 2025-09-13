import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm
from itertools import product
import pickle
import random
import open3d as o3d

PLT_DPI = 100

class OpticalFlowVisualizer:
    def __init__(self, viz_params, output, fps):
        self.params = viz_params
        if not self.params.viz_video: return

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        self.output_shape = self.params.viz_flags.shape
        self.output_file = output
        self.video_writer = cv.VideoWriter(output, fourcc, fps, self.params.vid_dims)

    def write_batch_frames(self, image, depth, dynamic_mask, orig_dynamic_masks, raft_flow, geom_flow, residual):
        if not self.params.viz_video: return
        
        print(f"\nWriting frames...\n")
        for frame in range(len(image)):
            frame_canvas = np.zeros((self.params.vid_dims[1], self.params.vid_dims[0], 3), dtype=np.uint8)

            for i, j in product(range(self.output_shape[0]), range(self.output_shape[1])):
                name = self.params.viz_flag_names[self.params.viz_flags[i][j]]

                x_offset = j * (self.params.vid_dims[0] // self.output_shape[1])
                y_offset = i * (self.params.vid_dims[1] // self.output_shape[0])
                width = self.params.vid_dims[0] // self.output_shape[1]
                height = self.params.vid_dims[1] // self.output_shape[0]

                if name == 'image':
                    img = image[frame]
                    resized_img = cv.resize(img, (width, height))
                    frame_canvas[y_offset:y_offset + height, x_offset:x_offset + width] = resized_img
                elif name == 'depth':
                    depth_colored = cv.applyColorMap(cv.normalize(depth[frame], None, 0, 255, cv.NORM_MINMAX).astype(np.uint8), cv.COLORMAP_JET)
                    resized_depth = cv.resize(depth_colored, (width, height))
                    frame_canvas[y_offset:y_offset + height, x_offset:x_offset + width] = resized_depth
                elif name == 'dynamic mask':
                    dynamic_colored = cv.cvtColor((dynamic_mask[frame] * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
                    resized_dynamic = cv.resize(dynamic_colored, (width, height))
                    frame_canvas[y_offset:y_offset + height, x_offset:x_offset + width] = resized_dynamic
                elif name == 'orig dynamic mask':
                    orig_dynamic_colored = cv.cvtColor((orig_dynamic_masks[frame] * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
                    resized_orig_dynamic = cv.resize(orig_dynamic_colored, (width, height))
                    frame_canvas[y_offset:y_offset + height, x_offset:x_offset + width] = resized_orig_dynamic
                elif name == 'raft flow':
                    flow_image = OpticalFlowVisualizer.viz_optical_flow_img(raft_flow[frame])
                    resized_flow = cv.resize(flow_image, (width, height))
                    frame_canvas[y_offset:y_offset + height, x_offset:x_offset + width] = resized_flow
                elif name == 'geometric flow':
                    flow_image = OpticalFlowVisualizer.viz_optical_flow_img(geom_flow[frame])
                    resized_flow = cv.resize(flow_image, (width, height))
                    frame_canvas[y_offset:y_offset + height, x_offset:x_offset + width] = resized_flow
                elif name == 'residual':
                    residual_normalized = np.clip(residual[frame] / self.params.viz_max_residual_magnitude, 0, 1)
                    residual_colored = cv.applyColorMap((residual_normalized * 255).astype(np.uint8), cv.COLORMAP_HOT)
                    resized_residual = cv.resize(residual_colored, (width, height))
                    frame_canvas[y_offset:y_offset + height, x_offset:x_offset + width] = resized_residual

            self.video_writer.write(frame_canvas)

    def end(self):
        if not self.params.viz_video: return
        print(f'video saved to {self.output_file}')
        self.video_writer.release()


    @classmethod
    def viz_optical_flow_img(cls, flow):
        hsv = np.zeros((*flow.shape[:-1], 3), dtype=np.uint8)
        valid_mask = ~np.isnan(flow[..., 0]) & ~np.isnan(flow[..., 1])

        magnitude, angle = cv.cartToPolar(flow[..., 0][valid_mask], flow[..., 1][valid_mask])

        hsv[..., 0][valid_mask] = (angle * 180 / np.pi / 2).flatten() # Hue corresponds to direction. OpenCV convention is 0-179
        hsv[..., 1][valid_mask] = 255  # Saturation is set to maximum
        hsv[..., 2][valid_mask] = (cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)).flatten()  # Value corresponds to magnitude
        hsv[..., 2][~valid_mask] = 255
        return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
   


def viz_tracked_objects(pickle_file, downsample_voxel_size=0.5, ids=None):
    with open(pickle_file, 'rb') as file:
        object_list = pickle.load(file)

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer()
    vis.show_skybox(False)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    vis.add_geometry("origin", origin)

    for id_, obj in object_list.items():
        if ids is not None and id_ not in ids: continue
        color = [random.random() for _ in range(3)]

        points = np.concatenate(obj['points'], axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
        pcd.paint_uniform_color(color)

        if len(pcd.points) < 3: continue

        centroid = points.mean(axis=0)
        max_z = points[:, 2].max()
        label_pos = centroid + np.array([0, 0, max_z - centroid[2] + 0.5])

        vis.add_geometry(f"pcl {id_}", pcd)
        vis.add_3d_label(label_pos, f"{id_}")

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()



@DeprecationWarning
def viz_optical_flow(flow):
    flow_image = OpticalFlowVisualizer.viz_optical_flow_img(flow)

    plt.figure(figsize=(10, 10))
    plt.imshow(flow_image)
    plt.title("Optical Flow")
    plt.axis("off")
    plt.show()

@DeprecationWarning
def viz_optical_flow_diff(magnitude_diff, angle_diff):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(magnitude_diff, cmap='hot')
    plt.title("Magnitude Difference")

    plt.colorbar(label="Scale")

    plt.subplot(1, 2, 2)
    plt.imshow(angle_diff, cmap='hsv')
    plt.title("Angle Difference")

    plt.show()

@DeprecationWarning
def viz_optical_flow_diff_batch(N, geometric_flow_batch, raft_flow_batch, image_batch, magnitude_diff_batch, norm_magnitude_diff_batch, angle_diff_batch, fps, 
                                output='optical_flow_diff.avi', OUT_WIDTH_RATIO=2.5, OUT_HEIGHT_RATIO=1.5):
    height, width = magnitude_diff_batch[0].shape
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    vid_size = (int(width * OUT_WIDTH_RATIO), int(height * OUT_HEIGHT_RATIO))
    out = cv.VideoWriter(output, fourcc, fps, vid_size)

    print(f'saving optical flow difference video to {output}...')

    figsize = (vid_size[0] / PLT_DPI, vid_size[1] / PLT_DPI)

    for i in tqdm(range(N)):
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        axes[0][0].imshow(image_batch[i])
        axes[0][0].set_title("Image")

        im1 = axes[0][1].imshow(magnitude_diff_batch[i], cmap='hot')
        axes[0][1].set_title("Magnitude Difference")
        cbar1 = fig.colorbar(im1, ax=axes[0][1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.set_label("Scale")

        im2 = axes[0][2].imshow(norm_magnitude_diff_batch[i], cmap='hot')
        axes[0][2].set_title("Normalized Magnitude Difference")
        cbar2 = fig.colorbar(im2, ax=axes[0][2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.set_label("Scale")

        im3 = axes[1][0].imshow(angle_diff_batch[i], cmap='hsv')
        axes[1][0].set_title("Angle Difference")
        cbar3 = fig.colorbar(im3, ax=axes[1][0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar3.set_label("Angle")

        axes[1][1].imshow(OpticalFlowVisualizer.viz_optical_flow_img(geometric_flow_batch[i]))
        axes[1][1].set_title("Geometric Optical Flow")

        axes[1][2].imshow(OpticalFlowVisualizer.viz_optical_flow_img(raft_flow_batch[i]))
        axes[1][2].set_title("RAFT Optical Flow")

        fig.tight_layout() 
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame[:, :, [1, 2, 3]]  # Convert ARGB to RGB by dropping the alpha channel
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        frame = cv.resize(frame, vid_size)
        out.write(frame)

        plt.close(fig)

    print('video saved successfully')

    out.release()