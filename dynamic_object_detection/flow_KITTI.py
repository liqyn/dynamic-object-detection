import torch


class GeometricOpticalFlow:
    def __init__(self, depth_camera_params, device: str = 'cuda'):
        self.device = device
        self.K = torch.tensor(depth_camera_params.K, dtype=torch.float32, device=device)                    # (3, 3)
        self.H = depth_camera_params.height
        self.W = depth_camera_params.width

        self.inv_K = torch.linalg.inv(self.K)  # (3, 3)

        self.pixel_coords, self.pixel_coords_flattened = self.compute_pixel_coords()                        # (H, W, 2), (H*W, 2)
        self.norm_cam_coords_h = self.get_norm_cam_coords_h(self.pixel_coords_flattened).view(1, -1, 3)     # (1, H*W, 3)

    def compute_flow(self, raft_flow, depth_images, T_1_0, use_3d=False):
        """
        raft_flow: (N, H, W, 2)
        depth_images: (N+1, H, W)
        T_1_0: (N, 4, 4)

        returns:
            coords_3d: (N, H*W, 3)
            resid_vel: (N, H, W)
        """

        raft_coords_3d_1 = self.raft_unproject(raft_flow, depth_images[1:])                                 # (N, H*W, 3), (N, H, W)

        if use_3d: 
            coords_3d, residual = self.compute_residual_3d_flow(raft_coords_3d_1, depth_images[:-1], T_1_0)
            geom_flow = None
        else: coords_3d, residual, geom_flow = self.compute_residual_2d_flow(raft_flow, depth_images[:-1], T_1_0)

        return residual, coords_3d, raft_coords_3d_1, geom_flow


    # --------------- For 2D flow --------------- #

    def compute_residual_2d_flow(self, raft_flow, depth_images, T_1_0):
        """
        raft_flow: (N, H, W, 2)
        depth_images: (N, H, W) (starts at index 0 of original depth images)
        T_1_0: (N, 4, 4)
        """
        coords_3d, gflow = self.compute_optical_flow_batch(depth_images, T_1_0)                             # (N, H, W, 2)

        resid_flow = raft_flow - gflow                                                                      # (N, H, W, 2)

        resid_flow = self.concat_last_axis(resid_flow, ones=False)                                          # (N, H, W, 3)

        resid_flow = torch.einsum('ij,nhwj->nhwi', self.inv_K, resid_flow)                                  # (N, H, W, 3)

        resid_vel = depth_images * torch.linalg.norm(resid_flow, dim=-1)                                    # (N, H, W)

        # not time normalized, done in tracker.py
        return coords_3d, resid_vel, gflow                                                                         # (N, H*W, 3), (N, H, W)                   
        

    @torch.no_grad()
    def compute_optical_flow_batch(self, depth_images, T_1_0):
        """
        depth_images: (N, H, W)
        T_1_0: (N, 4, 4)
        """
        N = len(depth_images)
        assert(N == len(T_1_0))

        invalid_mask = depth_images <= 0                                                                    # (N, H, W)

        coords_3d = self.unproject(depth_images)                                                            # (N, H*W, 3)
  
        coords_3d_1 = self.transform_next_frame(coords_3d, T_1_0)                                           # (N, H*W, 3)

        # project 3d coordinates in subsequent frame to original image frame
        projected_pixel_coords_flattened = self.project_points(coords_3d_1)                                 # (N, H*W, 2)

        # compute flow for each pixel in original frames
        flattened_flow = projected_pixel_coords_flattened - self.pixel_coords_flattened[None, :]            # (N, H*W, 2) - (1, H*W, 2) =   (N, H*W, 2)        

        flow = flattened_flow.view(N, self.H, self.W, 2)                                                    # (N, H, W, 2) 

        flow[invalid_mask] = torch.nan

        return coords_3d, flow                                                                              # (N, H*W, 3), (N, H, W, 2)   


    # --------------- For 3D flow --------------- #

    @torch.no_grad()
    def compute_residual_3d_flow(self, raft_coords_3d_1, depth_images, T_1_0):
        """
        raft_flow: (N, H, W, 2)
        depth_images: (N, H, W) (starts at index 0 of original depth images)
        T_1_0: (N, 4, 4)
        """
        N = len(raft_coords_3d_1)
        assert(N == len(depth_images) == len(T_1_0))

        coords_3d, geometric_coords_3d_1 = self.unproject_and_transform(depth_images, T_1_0)                # (N, H*W, 3) 
        
        residual = (raft_coords_3d_1 - geometric_coords_3d_1).view(N, self.H, self.W, 3)                    # (N, H, W, 3)

        return coords_3d, torch.linalg.norm(residual, dim=-1)                                               # (N, H*W, 3), (N, H, W)

    @torch.no_grad()
    def raft_unproject(self, raft_flow, subsq_depth_images):
        """
        raft_flow: (N, H, W, 2)
        subsq_depth_images: (N, H, W) (starts at index 1 of original depth images)
        """
        N = len(raft_flow)
        pixel_coords_after_flow_raw = raft_flow + self.pixel_coords[None, :]                                # (N, H, W, 2) + (1, H, W, 2) = (N, H, W, 2)

        pixel_coords_after_flow = torch.round(pixel_coords_after_flow_raw).to(torch.int)                    # (N, H, W, 2)
        invalid_mask = (pixel_coords_after_flow[..., 0] < 0) | (pixel_coords_after_flow[..., 0] > self.W - 1) | \
                       (pixel_coords_after_flow[..., 1] < 0) | (pixel_coords_after_flow[..., 1] > self.H - 1)                            # (N, H, W)
        
        flow_x = pixel_coords_after_flow[..., 0].clamp(0, self.W - 1)                                       # (N, H, W)
        flow_y = pixel_coords_after_flow[..., 1].clamp(0, self.H - 1)                                       # (N, H, W)
        batch_indices = torch.arange(N, dtype=torch.int, device=self.device).view(-1, 1, 1).expand(-1, self.H, self.W)                   # (N, H, W)

        depths_after_flow = subsq_depth_images[batch_indices, flow_y, flow_x]                               # (N, H, W)
        invalid_mask = invalid_mask | (depths_after_flow <= 0)                                              # (N, H, W)
        depths_after_flow_flattened = depths_after_flow.view(N, -1, 1)                                      # (N, H*W, 1)

        pixel_coords_after_flow_flattened = pixel_coords_after_flow_raw.reshape(-1, 2)                      # (N*H*W, 2) - raft_flow is non-contiguous due to permute

        new_norm_cam_coords_h = self.get_norm_cam_coords_h(pixel_coords_after_flow_flattened).view(N, -1, 3)                             # (N, H*W, 3)
        
        raft_coords_3d_1 = new_norm_cam_coords_h * depths_after_flow_flattened                              # (N, H*W, 3) * (N, H*W, 1) -> (N, H*W, 3)

        raft_coords_3d_1[invalid_mask.view(N, -1)] = torch.nan

        return raft_coords_3d_1                                                                             # (N, H*W, 3)
        
    @torch.no_grad()
    def unproject_and_transform(self, depth_images, T_1_0):
        """
        depth_images: (N, H, W)
        T_1_0: (N, 4, 4)
        """ 
        coords_3d = self.unproject(depth_images)                                                            # (N, H*W, 3)
        coords_3d_1 = self.transform_next_frame(coords_3d, T_1_0)                                           # (N, H*W, 3)
        return coords_3d, coords_3d_1                                                                       # (N, H*W, 3) x 2

    # --------------- Shared functions --------------- #

    def compute_pixel_coords(self):
        y, x = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32, device=self.device),
            torch.arange(self.W, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        pixel_coords = torch.stack((x, y), dim=-1)                                                          # (H, W, 2)
        pixel_coords_flattened = pixel_coords.view(-1, 2)                                                   # (H*W, 2)

        return pixel_coords, pixel_coords_flattened
    
    def get_norm_cam_coords_h(self, pixel_coords_flattened):
        """
        pixel_coords_flattened: (X, 2)
        """
        pixel_coords_flattened_h = self.concat_last_axis(pixel_coords_flattened)                            # (X, 3)
        norm_cam_coords = (self.inv_K @ pixel_coords_flattened_h.T).T[..., :2]                              # (X, 2)
        norm_cam_coords_h = self.concat_last_axis(norm_cam_coords)                                          # (X, 3)
        return norm_cam_coords_h

    def unproject(self, depth_images):
        """
        depth_images: (N, H, W)
        """
        flattened_depths = depth_images.view(depth_images.shape[0], -1, 1)                                  # (N, H*W, 1)
        coords_3d = self.norm_cam_coords_h * flattened_depths                                               # (1, H*W, 3) * (N, H*W, 1) -> (N, H*W, 3)
        coords_3d[flattened_depths.squeeze(-1) <= 0] = torch.nan
        return coords_3d                                                                                    # (N, H*W, 3)
    
    def project_points(self, coords_3d):
        """
        coords_3d: (N, H*W, 3)
        """
        coords_3d_norm = coords_3d / coords_3d[..., 2].unsqueeze(-1)
        pixel_coords = self.K @ coords_3d_norm.permute(0, 2, 1)                                             # (3, 3) * (N, 3, H*W) -> (N, 3, H*W)
        pixel_coords = pixel_coords.permute(0, 2, 1)                                                        # (N, H*W, 3)
        return pixel_coords[..., :2]                                                                        # (N, H*W, 2)
    
    def transform_next_frame(self, coords_3d, T_1_0):
        """
        coords_3d: (N, H*W, 3)
        T_1_0: (N, 4, 4)
        """
        coords_3d_h = self.concat_last_axis(coords_3d)                                                      # (N, H*W, 4)         
        coords_3d_h_1 = T_1_0 @ coords_3d_h.permute(0, 2, 1)                                                # (N, 4, 4) @ (N, 4, H*W) =     (N, 4, H*W)
        return (coords_3d_h_1.permute(0, 2, 1))[..., :3]                                                    # (N, 4, H*W) -> (N, H*W, 4) -> (N, H*W, 3)      

    def concat_last_axis(self, tensor, ones=True):
        if ones: return torch.concatenate((tensor, torch.ones((*tensor.shape[:-1], 1), dtype=torch.float32, device=self.device)), dim=-1)
        else: return torch.concatenate((tensor, torch.zeros((*tensor.shape[:-1], 1), dtype=torch.float32, device=self.device)), dim=-1)




    # --------------- Deprecated --------------- #

    # @DeprecationWarning
    # @classmethod
    # def compute_batched_flow_difference(cls, raft_batch, geometric_batch):
    #     flow_diff = raft_batch - geometric_batch                                                            # (N, H, W, 2)
    #     valid_mask = ~np.isnan(flow_diff[..., 0]) & ~np.isnan(flow_diff[..., 1])                            # (N, H, W)
    #     magnitude_diff, angle_diff = np.zeros((*flow_diff.shape[:-1], 1), dtype=np.float32), np.zeros((*flow_diff.shape[:-1], 1), dtype=np.float32)   # (N, H, W, 1) x 2
    #     magnitude_diff[valid_mask], angle_diff[valid_mask] = cv.cartToPolar(flow_diff[..., 0][valid_mask], flow_diff[..., 1][valid_mask])

    #     magnitude_diff[~valid_mask] = np.nan
    #     angle_diff[~valid_mask] = np.nan
    #     norm_magnitude_diff = magnitude_diff / np.linalg.norm(geometric_batch, axis=-1, keepdims=True)  # (N, H, W, 1)

    #     return magnitude_diff[..., 0], norm_magnitude_diff[..., 0], angle_diff[..., 0]  # (N, H, W) x 3

    # @DeprecationWarning
    # @torch.no_grad()
    # def compute_batched_flow_difference_torch(self, raft_batch, geometric_batch):
    #     raft_batch = torch.Tensor(raft_batch).to(self.device)                                              # (N, H, W, 2)
    #     geometric_batch = torch.Tensor(geometric_batch).to(self.device)                                    # (N, H, W, 2)

    #     flow_diff = raft_batch - geometric_batch                                                           # (N, H, W, 2)
    #     valid_mask = ~torch.isnan(flow_diff[..., 0]) & ~torch.isnan(flow_diff[..., 1])                     # (N, H, W)
    #     magnitude_diff, norm_magnitude_diff, angle_diff = (torch.zeros((*flow_diff.shape[:-1], 1), dtype=torch.float32, device=self.device) for _ in range(3))   # (N, H, W, 1) x 2
        
    #     magnitude_diff[~valid_mask] = torch.nan
    #     angle_diff[~valid_mask] = torch.nan
    #     magnitude_diff[valid_mask], angle_diff[valid_mask] = torch.sqrt(flow_diff[..., 0][valid_mask]**2 + flow_diff[..., 1][valid_mask]**2).view(-1, 1), \
    #                                                             (torch.atan2(flow_diff[..., 1][valid_mask], flow_diff[..., 0][valid_mask]) + torch.pi).view(-1, 1)

    #     norm_magnitude_diff = magnitude_diff / torch.linalg.norm(geometric_batch, dim=-1, keepdim=True)  # (N, H, W, 1)

    #     return tuple(ret[..., 0].cpu().numpy() for ret in (magnitude_diff, norm_magnitude_diff, angle_diff))  # (N, H, W) x 3


