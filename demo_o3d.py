import os
import glob
import argparse
from time import time
import torch.nn.functional as F
import numpy as np
import torch
import open3d as o3d
import random
from natsort import natsorted

from flashvggt.models.flash_vggt import FlashVGGT
from flashvggt_stream.models.flash_vggt import FlashVGGTStream
from flashvggt.utils.load_fn import load_and_preprocess_images
from flashvggt.utils.geometry_cuda import unproject_depth_map_to_point_map, normalize_camera_extrinsics_and_points_batch
from flashvggt.utils.pose_enc import pose_encoding_to_extri_intri

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="FlashVGGT demo with o3d for 3D visualization")
    parser.add_argument(
        "--model",
        type=str,
        default="FlashVGGT",
        help="Model to use: FlashVGGT, FlashVGGTStream",
    )
    parser.add_argument("--flash_ckpt_path", type=str, default="./ckpts/flashvggt.pt")
    parser.add_argument("--flash_stream_ckpt_path", type=str, default="./ckpts/flashvggt_stream.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Path to output directory")
    parser.add_argument(
        "--image_folder", 
        type=str, 
        default="./examples/garden/"
    )
    parser.add_argument("--max_points", type=int, default=1_000_000, help="Maximum number of points to visualize")
    parser.add_argument(
        "--conf_threshold", type=float, default=40.0, help="Initial percentage of low-confidence points to filter out"
    )
    parser.add_argument("--sample_rate", type=int, default=1, help="Sample rate of images to process")
    parser.add_argument("--max_images", type=int, default=800, help="Maximum number of images to process")
    parser.add_argument("--mode", type=str, default="crop", help="Mode to process images")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10,
        help="Frame chunk size for FlashVGGTStream streaming inference (forward chunk_size)",
    )
    parser.add_argument("--kv_downfactor", type=int, default=3, help="KV downfactor for FlashVGGT")
    parser.add_argument("--keyframe_every", type=int, default=200, help="Keyframe interval for FlashVGGT")
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Device: {device}, Dtype: {dtype}")

    print("Initializing and loading FlashVGGT model...")
    # set up models
    if args.model == "FlashVGGT":
        model = FlashVGGT( kv_downfactor=args.kv_downfactor, keyframe_every=args.keyframe_every)
        model.load_ckpt(args.flash_ckpt_path)
    elif args.model == "FlashVGGTStream":
        model = FlashVGGTStream(kv_downfactor=args.kv_downfactor)
        model.load_ckpt(args.flash_stream_ckpt_path)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    model.eval()
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = glob.glob(os.path.join(args.image_folder, "*" ))
    image_names = natsorted(image_names)
    image_names = image_names[::args.sample_rate][:args.max_images]
    print(f"Found {len(image_names)} images")
    num_images = len(image_names)

    images = load_and_preprocess_images(image_names, mode=args.mode) # [N, 3, H, W]
    if torch.is_tensor(images):
        images = images.to(device)
    else:
        images = torch.from_numpy(images).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    start_time = time()
    with torch.amp.autocast("cuda", dtype=dtype):
        if args.model == "FlashVGGTStream":
            predictions = model(images, chunk_size=args.chunksize)
        else:
            predictions = model(images)
    inference_time = time() - start_time
    print(f"Inference time: {inference_time:.2f}s")
    print(f"GPU max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Generating point cloud...")

    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            if predictions[key].shape[0] == 1:
                predictions[key] = predictions[key].squeeze(0).to(device).float()

    # Unpack prediction dict
    images = predictions["images"]  # (S, 3, H, W)
    depth_map = predictions["depth"]  # (S, H, W, 1)
    depth_conf = predictions["depth_conf"]  # (S, H, W)
    extrinsics_cam = predictions["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = predictions["intrinsic"]  # (S, 3, 3)

    world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

    extrinsics, _, world_points, depth_map = \
        normalize_camera_extrinsics_and_points_batch(
            extrinsics=extrinsics_cam.unsqueeze(0),
            cam_points=torch.zeros_like(world_points.unsqueeze(0)), # won't be used
            world_points=world_points.unsqueeze(0),
            depths=depth_map.unsqueeze(0),
            point_masks=torch.ones_like(depth_map.unsqueeze(0)), # use the same point masks as the ground truth
            scale_by_points=True,
        )
    world_points = world_points.squeeze(0)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.permute(0, 2, 3, 1)  # now (S, H, W, 3)
    points = world_points.reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3)
    conf_flat = depth_conf.reshape(-1)

    if points.shape[0] > args.max_points:
        indices = torch.randperm(len(points))[:args.max_points].to(device)
        points = points[indices]
        colors_flat = colors_flat[indices]
        conf_flat = conf_flat[indices]

    # Filter points based on confidence
    if args.conf_threshold > 0.0:
        threshold_val = torch.quantile(conf_flat, args.conf_threshold / 100.0)
        conf_mask = (conf_flat > threshold_val) & (conf_flat > 0.1)
        points = points[conf_mask]
        colors_flat = colors_flat[conf_mask]
        conf_flat = conf_flat[conf_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors_flat.detach().cpu().numpy())

    ply_path = os.path.join(args.output_dir, f"{args.model}_{num_images}images_{inference_time:.2f}s.ply")
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved {len(points)} points to {ply_path} in {inference_time:.2f}s for {num_images} images")

if __name__ == "__main__":
    main()