import torch

# Pre-create large enough grid coordinate tensors, assuming a maximum image size of 2048x2048
_MAX_SIZE = 2048
_GRID_U = torch.arange(_MAX_SIZE).reshape(1, -1).repeat(_MAX_SIZE, 1)  # Duplicate rows
_GRID_V = torch.arange(_MAX_SIZE).reshape(-1, 1).repeat(1, _MAX_SIZE)  # Duplicate columns
_GRID_U = _GRID_U.float()
_GRID_V = _GRID_V.float()

def deproject_numpy(intrinsic_cv, cam2world_gl, depth_img):
    pcd = deproject(
        torch.from_numpy(intrinsic_cv),
        torch.from_numpy(cam2world_gl),
        torch.from_numpy(depth_img),
    )
    return pcd.numpy()

def deproject(intrinsic_cv, cam2world_gl, depth_img):
    """
    Deprojects pixel points to 3D coordinates using Torch.

    Args:
        intrinsic_cv: (3, 3) torch.Tensor; Camera intrinsic matrix according to opencv definition.
                   [[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]]
        cam2world_gl: (4, 4) torch.Tensor; Camera-to-world transformation matrix (including rotation and translation).
                        The camera coordinate frame is according to opengl definition.
        depth_img: (H, W) torch.Tensor; Depth image, used as a reference to generate 3D coordinates.

    Returns:
        (3, H, W) torch.Tensor; World coordinates (x, y, z) of the deprojected points. 
    """
    h, w = depth_img.shape
    
    # 1. Use pre-created grids and slice them according to the actual image size
    u = _GRID_U[:h, :w].reshape(-1)  # (H*W)
    v = _GRID_V[:h, :w].reshape(-1)  # (H*W)
    
    # 2. Get the depth values
    z = depth_img.reshape(-1)  # (H*W)

    # 3. Deproject using the intrinsic_cv matrix
    ones = torch.ones_like(z)
    pixel_coords = torch.stack([u, v, ones], dim=0)  # (3, H*W)
    intrinsic_cv_inv = torch.inverse(intrinsic_cv)
    cam_coords = intrinsic_cv_inv @ pixel_coords * z  # (3, H*W)  (Handles division by fx, fy, and subtraction of cx, cy)

    # 4. From OpenCV to OpenGL coordinate system
    cam_coords_opengl = cam_coords.clone()
    cam_coords_opengl[1, :] = -cam_coords_opengl[1, :]  
    cam_coords_opengl[2, :] = -cam_coords_opengl[2, :] 

    # 5. Transform to world coordinates
    cam_coords_opengl = torch.cat([cam_coords_opengl, ones.unsqueeze(0)], dim=0)  # (4, H*W)
    world_coords = cam2world_gl @ cam_coords_opengl  # (4, H*W)

    return world_coords[:3].reshape(3, h, w)