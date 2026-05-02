import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from omegaconf import OmegaConf


FILE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = FILE_DIR.parent
REPO_ROOT = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from data.datasets.blendedmvs import BlendedMVSDataset  # noqa: E402
from data.datasets.mapillary import MapillaryDataset  # noqa: E402
from data.datasets.mvs_synth import MVSSynthDataset  # noqa: E402
from data.datasets.scannet import ScanNetDataset  # noqa: E402
from data.datasets.vkitti import VKittiDataset  # noqa: E402


def build_common_config(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "img_size": args.img_size,
            "patch_size": args.patch_size,
            "rescale": args.rescale,
            "rescale_aug": args.rescale_aug,
            "landscape_check": args.landscape_check,
            "debug": args.debug,
            "training": args.training,
            "get_nearby": args.get_nearby,
            "inside_random": False,
            "allow_duplicate_img": args.allow_duplicate_img,
            "augs": {
                "scales": args.aug_scales,
            },
        }
    )


def _extract_valid_points_and_colors(
    world_points: np.ndarray,
    image: np.ndarray,
    point_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if world_points.ndim != 3 or world_points.shape[-1] != 3:
        raise ValueError(f"Expected world_points to have shape (H, W, 3), got {world_points.shape}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected image to have shape (H, W, 3), got {image.shape}")
    if point_mask.shape != world_points.shape[:2]:
        raise ValueError(
            f"Mask shape {point_mask.shape} does not match point shape {world_points.shape[:2]}"
        )

    valid_mask = point_mask.astype(bool)
    finite_mask = np.isfinite(world_points).all(axis=-1)
    valid_mask = valid_mask & finite_mask

    points = world_points[valid_mask].reshape(-1, 3).astype(np.float64)
    colors = image[valid_mask].reshape(-1, 3).astype(np.float64)
    if colors.size > 0 and colors.max() > 1.0:
        colors = colors / 255.0
    return points, colors


def save_masked_world_points_as_ply(
    world_points_list: list[np.ndarray],
    image_list: list[np.ndarray],
    point_mask_list: list[np.ndarray],
    output_path: Path,
    max_points: int,
) -> int:
    if not (len(world_points_list) == len(image_list) == len(point_mask_list)):
        raise ValueError("world_points_list, image_list, and point_mask_list must have the same length.")
    if len(world_points_list) == 0:
        raise RuntimeError("No frames were provided for point cloud export.")

    point_chunks = []
    color_chunks = []
    for world_points, image, point_mask in zip(world_points_list, image_list, point_mask_list):
        points, colors = _extract_valid_points_and_colors(world_points, image, point_mask)
        if points.shape[0] > 0:
            point_chunks.append(points)
            color_chunks.append(colors)

    if len(point_chunks) == 0:
        raise RuntimeError("No valid points found after applying point/finite masks.")

    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)

    if max_points > 0 and points.shape[0] > max_points:
        sampled_idx = np.random.choice(points.shape[0], size=max_points, replace=False)
        points = points[sampled_idx]
        colors = colors[sampled_idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    ok = o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=True)
    if not ok:
        raise RuntimeError(f"Failed to write point cloud to {output_path}")
    return points.shape[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample a training dataset and export world point clouds from all sampled frames to .ply."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="vkitti",
        choices=["vkitti", "blendedmvs", "mvs_synth", "scannet", "mapillary"],
        help="Dataset backend to use.",
    )
    parser.add_argument(
        "--vkitti-dir",
        type=str,
        default="/home/zwang253/workspace/scale-up-vggt/Key3R/data/training/vkitti",
        help="Path to VKitti root directory.",
    )
    parser.add_argument(
        "--blendedmvs-dir",
        type=str,
        default="/home/zwang253/workspace/scale-up-vggt/Key3R/data/training/blendedmvs",
        help="Path to BlendedMVS root directory.",
    )
    parser.add_argument(
        "--mvs-synth-dir",
        type=str,
        default="/home/zwang253/workspace/scale-up-vggt/Key3R/data/training/mvs_synth",
        help="Path to MVS-Synth root directory.",
    )
    parser.add_argument(
        "--scannet-dir",
        type=str,
        default="/home/zwang253/workspace/scale-up-vggt/Key3R/data/training/scannet/full_dataset",
        help="Path to ScanNet full_dataset root directory.",
    )
    parser.add_argument(
        "--mapillary-dir",
        type=str,
        default="/home/zwang253/workspace/scale-up-vggt/Key3R/data/training/mapillary",
        help="Path to Mapillary root directory.",
    )
    parser.add_argument("--seq-index", type=int, default=0, help="Sequence index in VKitti list.")
    parser.add_argument("--img-per-seq", type=int, default=8, help="Number of images sampled per sequence.")
    parser.add_argument("--aspect-ratio", type=float, default=1.0)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--min-num-images", type=int, default=24)
    parser.add_argument("--len-train", type=int, default=100000)
    parser.add_argument("--expand-ratio", type=int, default=8)
    parser.add_argument("--training", action="store_true", help="Enable dataset training mode augmentations.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--get-nearby", action="store_true", help="Use nearby frame sampling.")
    parser.add_argument("--allow-duplicate-img", action="store_true", help="Allow duplicated sampled IDs.")
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rescale-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--landscape-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--aug-scales",
        nargs=2,
        type=float,
        default=[0.8, 1.2],
        metavar=("MIN_SCALE", "MAX_SCALE"),
        help="Scale augmentation range for BaseDataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output .ply path.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500000,
        help="Randomly downsample if total points exceed this value. Use <=0 to disable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    common_conf = build_common_config(args)
    if args.dataset == "vkitti":
        dataset = VKittiDataset(
            common_conf=common_conf,
            VKitti_DIR=args.vkitti_dir,
            min_num_images=args.min_num_images,
            len_train=args.len_train,
            expand_ratio=args.expand_ratio,
        )
        dataset_dir = args.vkitti_dir
    elif args.dataset == "blendedmvs":
        dataset = BlendedMVSDataset(
            common_conf=common_conf,
            BLENDEDMVS_DIR=args.blendedmvs_dir,
            min_num_images=args.min_num_images,
            len_train=args.len_train,
            expand_ratio=args.expand_ratio,
        )
        dataset_dir = args.blendedmvs_dir
    elif args.dataset == "mvs_synth":
        dataset = MVSSynthDataset(
            common_conf=common_conf,
            MVSSYNTH_DIR=args.mvs_synth_dir,
            min_num_images=args.min_num_images,
            len_train=args.len_train,
            expand_ratio=args.expand_ratio,
        )
        dataset_dir = args.mvs_synth_dir
    elif args.dataset == "mapillary":
        dataset = MapillaryDataset(
            common_conf=common_conf,
            MAPILLARY_DIR=args.mapillary_dir,
            min_num_images=args.min_num_images,
            len_train=args.len_train,
            expand_ratio=args.expand_ratio,
        )
        dataset_dir = args.mapillary_dir
    else:
        dataset = ScanNetDataset(
            common_conf=common_conf,
            SCANNET_DIR=args.scannet_dir,
            min_num_images=args.min_num_images,
            len_train=args.len_train,
            expand_ratio=args.expand_ratio,
        )
        dataset_dir = args.scannet_dir

    if dataset.sequence_list_len == 0:
        raise RuntimeError(f"No valid sequences found for {args.dataset} under {dataset_dir}")

    if args.seq_index < 0 or args.seq_index >= dataset.sequence_list_len:
        raise IndexError(
            f"seq-index={args.seq_index} out of range [0, {dataset.sequence_list_len - 1}]"
        )

    batch = dataset.get_data(
        seq_index=args.seq_index,
        img_per_seq=args.img_per_seq,
        aspect_ratio=args.aspect_ratio,
    )
    if len(batch["world_points"]) == 0:
        raise RuntimeError("Dataset returned empty world_points list.")

    out_path = (
        Path(args.output)
        if args.output
        else (REPO_ROOT / f"logs/analysis/datasets/{args.dataset}.ply")
    )
    num_points = save_masked_world_points_as_ply(
        world_points_list=batch["world_points"],
        image_list=batch["images"],
        point_mask_list=batch["point_masks"],
        output_path=out_path,
        max_points=args.max_points,
    )

    print(f"Saved {args.dataset} point cloud: {out_path}")
    print(f"Sequence: {batch['seq_name']}")
    print(f"Sampled ids: {batch['ids']}")
    print(f"Frame count used: {len(batch['world_points'])}")
    print(f"Max points: {args.max_points}")
    print(f"Point count: {num_points}")


if __name__ == "__main__":
    main()
