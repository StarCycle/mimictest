import torch

# Pre-create large enough grid coordinate tensors, assuming a maximum image size of 2048x2048
_MAX_SIZE = 2048
_GRID_U = torch.arange(_MAX_SIZE).reshape(1, -1).repeat(_MAX_SIZE, 1)  # Duplicate rows
_GRID_V = torch.arange(_MAX_SIZE).reshape(-1, 1).repeat(1, _MAX_SIZE)  # Duplicate columns
_GRID_U = _GRID_U.float()
_GRID_V = _GRID_V.float()

def deproject(intrinsic, T_world_cam, depth_img):
    """
    Deprojects pixel points to 3D coordinates using Torch.

    Args:
        intrinsic: (3, 3) torch.Tensor; Camera intrinsic matrix.
                   [[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]]
        T_world_cam: (4, 4) torch.Tensor; Camera-to-world transformation matrix (including rotation and translation).
        depth_img: (H, W) torch.Tensor; Depth image, used as a reference to generate 3D coordinates.

    Returns:
        (3, H, W) torch.Tensor; World coordinates (x, y, z) of the deprojected points. 
    """
    h, w = depth_img.shape
    
    # 1. Use pre-created grids and slice them according to the actual image size
    u = _GRID_U[:h, :w].reshape(-1)  # (H*W)
    v = _GRID_V[:h, :w].reshape(-1)  # (H*W)
    
    # 2. Get the depth values
    z = - depth_img.reshape(-1)  # (H*W)

    # 3. Deproject using the intrinsic matrix
    ones = torch.ones_like(z)
    pixel_coords = torch.stack([u, v, ones], dim=0)  # (3, H*W)
    intrinsic_inv = torch.inverse(intrinsic)
    cam_coords = intrinsic_inv @ pixel_coords * z  # (3, H*W)  (Handles division by fx, fy, and subtraction of cx, cy)
    cam_coords = torch.cat([cam_coords, ones.unsqueeze(0)], dim=0) # Add a dimension to make it (4, H*W) for homogeneous transformation

    # 4. Transform to world coordinates
    world_coords = T_world_cam @ cam_coords  # (4, H*W)

    return world_coords[:3].reshape(3, h, w)