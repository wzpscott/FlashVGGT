import torch
import logging
from typing import Optional, Tuple

def unproject_depth_map_to_point_map(
    depth_map: torch.Tensor, extrinsics_cam: torch.Tensor, intrinsics_cam: torch.Tensor
) -> torch.Tensor:
    """
    Unproject a batch of depth maps to 3D world coordinates using PyTorch.

    Args:
        depth_map (torch.Tensor): Batch of depth maps of shape (S, H, W, 1) or (S, H, W).
        extrinsics_cam (torch.Tensor): Batch of camera extrinsic matrices of shape (S, 3, 4).
        intrinsics_cam (torch.Tensor): Batch of camera intrinsic matrices of shape (S, 3, 3).

    Returns:
        torch.Tensor: Batch of 3D world coordinates of shape (S, H, W, 3).
    """
    if depth_map is None:
        return None

    if depth_map.dim() == 4 and depth_map.shape[-1] == 1:
        depth_map = depth_map.squeeze(-1)

    device = depth_map.device
    dtype = depth_map.dtype

    extrinsics_cam = extrinsics_cam.to(device=device, dtype=dtype)
    intrinsics_cam = intrinsics_cam.to(device=device, dtype=dtype)

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx], extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = torch.stack(world_points_list, dim=0)

    return world_points_array


def depth_to_world_coords_points(
    depth_map: torch.Tensor,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a depth map to world coordinates using PyTorch.

    Args:
        depth_map (torch.Tensor): Depth map of shape (H, W).
        intrinsic (torch.Tensor): Camera intrinsic matrix of shape (3, 3).
        extrinsic (torch.Tensor): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: World coordinates (H, W, 3), camera coordinates (H, W, 3), valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    device = depth_map.device
    dtype = depth_map.dtype
    extrinsic = extrinsic.to(device=device, dtype=dtype)
    intrinsic = intrinsic.to(device=device, dtype=dtype)

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = cam_coords_points @ R_cam_to_world.T + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask
	
def depth_to_cam_coords_points(depth_map: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """
    Convert a depth map to camera coordinates using PyTorch.

    Args:
        depth_map (torch.Tensor): Depth map of shape (H, W).
        intrinsic (torch.Tensor): Camera intrinsic matrix of shape (3, 3).

    Returns:
        torch.Tensor: Camera coordinates of shape (H, W, 3).
    """
    H, W = depth_map.shape
    assert intrinsic.shape[-2:] == (3, 3), "Intrinsic matrix must be 3x3"
    zero = torch.tensor(0.0, dtype=intrinsic.dtype, device=intrinsic.device)
    assert torch.allclose(intrinsic[0, 1], zero) and torch.allclose(intrinsic[1, 0], zero), "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    device = depth_map.device
    dtype = depth_map.dtype

    # Generate grid of pixel coordinates (match NumPy's meshgrid with indexing='xy')
    uu = torch.arange(W, device=device, dtype=dtype)
    vv = torch.arange(H, device=device, dtype=dtype)
    u, v = torch.meshgrid(uu, vv, indexing='xy')

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = torch.stack((x_cam, y_cam, z_cam), dim=-1)

    return cam_coords


def closed_form_inverse_se3(se3: torch.Tensor, R: torch.Tensor | None = None, T: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch using PyTorch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3 (torch.Tensor): Tensor of shape (N, 4, 4) or (N, 3, 4).
        R (torch.Tensor, optional): Tensor of shape (N, 3, 3).
        T (torch.Tensor, optional): Tensor of shape (N, 3, 1).

    Returns:
        torch.Tensor: Inverted SE3 matrices of shape (N, 4, 4).
    """
    if se3.shape[-2:] not in [(4, 4), (3, 4)]:
        raise ValueError(f"se3 must be of shape (N,4,4) or (N,3,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:].reshape(-1, 3, 1)  # (N,3,1)

    N = R.shape[0]
    R_transposed = R.transpose(1, 2)  # (N,3,3)
    top_right = -torch.bmm(R_transposed, T)  # (N,3,1)

    inverted_matrix = torch.eye(4, dtype=R.dtype, device=R.device).expand(N, 4, 4).clone()
    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    
    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")


@torch.no_grad()
def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_points: bool = True,
    point_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
    """
    # Validate inputs
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")


    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    # assert device == torch.device("cpu")

    # Convert extrinsics to homogeneous form: (B, N, 4, 4)
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)

    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None

    if scale_by_points:
        new_cam_points = cam_points.clone()
        new_depths = depths.clone()

        dist = new_world_points.norm(dim=-1)
        # dist_sum = (dist * point_masks.unsqueeze(-1)).sum(dim=[1,2,3])
        # valid_count = point_masks.sum(dim=[1,2,3])
        # avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)
        dist_median = torch.median(dist)
        avg_scale = dist_median

        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths

    new_extrinsics = new_extrinsics[:, :, :3] # 4x4 -> 3x4
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)

    return new_extrinsics, new_cam_points, new_world_points, new_depths


def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
    
    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                  [-hard_max, hard_max] will be clamped. If None, 
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor
    
    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor