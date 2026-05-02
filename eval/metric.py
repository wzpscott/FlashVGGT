import torch
import numpy as np
import evo
from evo.core import trajectory, metrics, sync
import copy

from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree as KDTree
import open3d as o3d

from flash_vggt.utils.pose_enc import pose_encoding_to_extri_intri
from flash_vggt.utils.geometry_cuda import unproject_depth_map_to_point_map

from data.normalization import normalize_camera_extrinsics_and_points_batch

class MultiMetric(torch.nn.Module):
    def __init__(self, metrics=["depth", "pose", "point"]):
        super().__init__()

        self.dtype = torch.float32

        self.metrics = []
        if "depth" in metrics:
            self.metrics.append(DepthMetric())
        if "pose" in metrics:
            self.metrics.append(PoseMetric())
        if "point" in metrics:
            self.metrics.append(PointMetric())

    def forward(self, predicted, ground_truth):
        predicted = self.process_predicted(predicted)
        predicted, ground_truth = self.shift_and_scale(predicted, ground_truth)

        predicted = self.remove_batch_dimension(predicted)
        ground_truth = self.remove_batch_dimension(ground_truth)

        results = {}
        for metric in self.metrics:
            results.update(metric(predicted, ground_truth))
        return results

    def process_predicted(self, predicted):
        predicted_pose = predicted["pose_enc"]
        if not torch.is_tensor(predicted_pose):
            predicted_pose = torch.from_numpy(predicted_pose)

        predicted_depth = predicted["depth"]
        if predicted_depth.shape[-1] == 1:
            predicted_depth = predicted_depth.squeeze(-1)
        if predicted_depth.shape[0] == 1:
            predicted_depth = predicted_depth.squeeze(0)
        
        image_size = predicted_depth.shape[-2:]
        predicted_extrinsic, predicted_intrinsic = pose_encoding_to_extri_intri(predicted_pose, image_size)

        predicted["extrinsics"] = predicted_extrinsic.to(self.dtype)
        predicted["intrinsics"] = predicted_intrinsic.to(self.dtype)

        if predicted_extrinsic.shape[0] == 1:
            predicted_extrinsic = predicted_extrinsic.squeeze(0)
        if predicted_intrinsic.shape[0] == 1:
            predicted_intrinsic = predicted_intrinsic.squeeze(0)

        predicted_point = unproject_depth_map_to_point_map(predicted_depth, predicted_extrinsic, predicted_intrinsic)

        predicted["world_coords_points"] = predicted_point.to(self.dtype)
        predicted["depth"] = predicted_depth.to(self.dtype)

        return predicted

    def shift_and_scale(self, predicted, ground_truth, max_iters=10):
        # shift the ground truth in the coordinate of the first frame
        ground_truth["extrinsics"], ground_truth["cam_coords_points"], ground_truth["world_coords_points"], ground_truth["depths"] = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=ground_truth["extrinsics"],
                cam_points=ground_truth["cam_coords_points"],
                world_points=ground_truth["world_coords_points"],
                depths=ground_truth["depths"],
                point_masks=ground_truth["point_masks"],
                scale_by_points=True,
            )
        
        predicted["extrinsics"], predicted["cam_coords_points"], predicted["world_coords_points"], predicted["depth"] = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=predicted["extrinsics"],
                cam_points=torch.zeros_like(ground_truth["cam_coords_points"]), # won't be used
                world_points=predicted["world_coords_points"],
                depths=predicted["depth"],
                point_masks=ground_truth["point_masks"], # use the same point masks as the ground truth
                scale_by_points=True,
            )

        # # find a scale factor to scale the predicted depth to the ground truth depth
        # scale = (torch.median(ground_truth["depths"]) / torch.median(predicted["depth"])).item()

        # for _ in range(max_iters):
        #     # compute the residuals
        #     residuals = scale * predicted["depth"] - ground_truth["depths"]
        #     # compute the weights
        #     weights = 1.0 / (residuals.abs() + 1e-8)
        #     # update the scale factor
        #     scale = (torch.sum(weights * predicted["depth"] * ground_truth["depths"]) / torch.sum(weights * predicted["depth"]**2)).item()
        
        # scale = max(scale, 1e-3)

        # predicted["depth"] = predicted["depth"] * scale
        # predicted["world_coords_points"] = predicted["world_coords_points"] * scale
        # predicted["extrinsics"][..., :3, 3] = predicted["extrinsics"][..., :3, 3] * scale

        return predicted, ground_truth

    def remove_batch_dimension(self, data):
        for key in data.keys():
            if torch.is_tensor(data[key]):
                if data[key].shape[0] == 1:
                    data[key] = data[key].squeeze(0)
        return data


class DepthMetric(torch.nn.Module):
    def __init__(
        self,
        max_depth=10,
        post_clip_min=None,
        post_clip_max=None,
        pre_clip_min=None,
        pre_clip_max=None,
        use_gpu=True,
    ):
        super().__init__()

        self.max_depth = max_depth
        self.post_clip_min = post_clip_min
        self.post_clip_max = post_clip_max
        self.pre_clip_min = pre_clip_min
        self.pre_clip_max = pre_clip_max
        self.use_gpu = use_gpu

    def forward(self, predicted, ground_truth, custom_mask=None):
        predicted_depth = predicted["depth"]
        ground_truth_depth = ground_truth["depths"]
        if predicted_depth.shape[-1] == 1:
            predicted_depth = predicted_depth.squeeze(-1)
        if ground_truth_depth.shape[-1] == 1:
            ground_truth_depth = ground_truth_depth.squeeze(-1)

        assert predicted_depth.shape == ground_truth_depth.shape, "Predicted depth and ground truth depth must have the same shape"
        assert predicted_depth.dim() == 3, "Predicted depth must have shape (B, H, W)"

        _, h, w = predicted_depth.shape
        predicted_depth = predicted_depth.view(-1, w)
        ground_truth_depth = ground_truth_depth.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)

        if self.use_gpu:
            predicted_depth = predicted_depth.cuda()
            ground_truth_depth = ground_truth_depth.cuda()
            if custom_mask is not None:
                custom_mask = custom_mask.cuda()

        # Filter out depths greater than max_depth
        depth_mask = ground_truth_depth > 0
        if self.max_depth is not None:
            depth_mask = torch.logical_and(depth_mask, ground_truth_depth < self.max_depth)
        if custom_mask is not None:
            depth_mask = torch.logical_and(depth_mask, custom_mask)
        predicted_depth = predicted_depth[depth_mask]
        ground_truth_depth = ground_truth_depth[depth_mask]

        # Clip the depth values
        if self.pre_clip_min is not None:
            predicted_depth = torch.clamp(predicted_depth, min=self.pre_clip_min)
        if self.pre_clip_max is not None:
            predicted_depth = torch.clamp(predicted_depth, max=self.pre_clip_max)

        # Clip the predicted depth values
        if self.post_clip_min is not None:
            predicted_depth = torch.clamp(predicted_depth, min=self.post_clip_min)
        if self.post_clip_max is not None:
            predicted_depth = torch.clamp(predicted_depth, max=self.post_clip_max)

        # Calculate the metrics
        abs_rel = torch.median(
            torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth
        ).item()

        # Clip the depth values to avoid division by zero
        predicted_depth = torch.clamp(predicted_depth, min=1e-5)

        # Calculate the accuracy thresholds
        max_ratio = torch.maximum(
            predicted_depth / ground_truth_depth, ground_truth_depth / predicted_depth
        )

        threshold = torch.mean((max_ratio < 1.25).float()).item()

        results = {
            "Depth-Rel.": abs_rel,
            "Depth-τ": threshold,
        }

        return results

class PoseMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.ate_metric = metrics.APE(metrics.PoseRelation.translation_part)
        self.are_metric = metrics.APE(metrics.PoseRelation.rotation_part)
        self.rpe_trans_metric = metrics.RPE(metrics.PoseRelation.translation_part)
        self.rpe_rot_metric = metrics.RPE(metrics.PoseRelation.rotation_part)
        
    def forward(self, predicted, ground_truth):
        predicted_pose = predicted["extrinsics"]
        ground_truth_pose = ground_truth["extrinsics"]

        traj_est = self.poses_to_evo_traj(predicted_pose)
        traj_gt = self.poses_to_evo_traj(ground_truth_pose)

        # align the trajectories
        traj_est.align(traj_gt, correct_scale=False, correct_only_scale=True)

        results = {}
        data = (traj_gt, traj_est)

        self.ate_metric.process_data(data)
        ate_stat = self.ate_metric.get_statistic(metrics.StatisticsType.rmse)
        results["Cam-APE"] = ate_stat

        self.rpe_trans_metric.process_data(data)
        rpe_trans_stat = self.rpe_trans_metric.get_statistic(metrics.StatisticsType.rmse)
        results["Cam-RPE-Trans"] = rpe_trans_stat

        self.are_metric.process_data(data)
        are_stat = self.are_metric.get_statistic(metrics.StatisticsType.rmse)
        results["Cam-ARE"] = are_stat

        self.rpe_rot_metric.process_data(data)
        rpe_rot_stat = self.rpe_rot_metric.get_statistic(metrics.StatisticsType.rmse)
        results["Cam-RPE-Rot"] = rpe_rot_stat

        return results

    def poses_to_evo_traj(self, poses_c2w, timestamps=None):
        """
        Convert [N, 4, 4] c2w pose matrices to evo trajectory format
        """
        positions = poses_c2w[:, :3, 3]  # Extract translation components
        rotations = poses_c2w[:, :3, :3]  # Extract rotation matrices

        if torch.is_tensor(positions):
            positions = positions.float().cpu().numpy()
        if torch.is_tensor(rotations):
            rotations = rotations.float().cpu().numpy()
        
        # If no timestamps provided, create dummy ones
        if timestamps is None:
            timestamps = np.arange(len(poses_c2w)).reshape(-1, 1).astype(np.float32)

        # Convert rotations to quaternions in w, x, y, z format
        quats = R.from_matrix(rotations).as_quat()  # [x, y, z, w] format
        quats_wxyz = np.roll(quats, shift=1, axis=1)  # [w, x, y, z] format for evo
        
        # Create trajectory object
        traj = trajectory.PoseTrajectory3D(
            positions_xyz=positions,
            orientations_quat_wxyz=quats_wxyz,
            timestamps=timestamps
        )
        
        return traj

class PointMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predicted, ground_truth):
        predicted_point = predicted["world_coords_points"].detach().cpu().numpy()
        ground_truth_point = ground_truth["world_coords_points"].detach().cpu().numpy()
        predicted_point, ground_truth_point = predicted_point.reshape(-1, 3), ground_truth_point.reshape(-1, 3)

        if predicted_point.shape[0] > int(1e6):
            sample_indices = np.random.choice(
                predicted_point.shape[0], int(1e6), replace=False
            )
            predicted_point = predicted_point[sample_indices]
            ground_truth_point = ground_truth_point[sample_indices]

        predicted_pcd = o3d.geometry.PointCloud()
        ground_truth_pcd = o3d.geometry.PointCloud()
        predicted_pcd.points = o3d.utility.Vector3dVector(predicted_point)
        ground_truth_pcd.points = o3d.utility.Vector3dVector(ground_truth_point)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            predicted_pcd,
            ground_truth_pcd,
            0.05,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        transformation = reg_p2p.transformation
        predicted_pcd = predicted_pcd.transform(transformation)

        # accuracy_dists = predicted_pcd.compute_point_cloud_distance(ground_truth_pcd)
        # accuracy = np.mean(np.asarray(accuracy_dists))

        # completion_dists = ground_truth_pcd.compute_point_cloud_distance(predicted_pcd)
        # completion = np.mean(np.asarray(completion_dists))

        # chamfer_distance = (accuracy + completion) / 2
        accuracy = self.accuracy(ground_truth_point, predicted_point)[1]
        completion = self.completion(ground_truth_point, predicted_point)[1]
        chamfer_distance = (accuracy + completion) / 2

        # Normal consistency
        predicted_pcd.estimate_normals()
        ground_truth_pcd.estimate_normals()
        predicted_normals = np.asarray(predicted_pcd.normals)
        ground_truth_normals = np.asarray(ground_truth_pcd.normals)
        normal_consistency = np.mean(np.abs(np.sum(predicted_normals * ground_truth_normals, axis=-1)))

        return {
            "Point-Acc.": accuracy,
            "Point-Comp.": completion,
            "Point-CD": chamfer_distance,
            "Point-NC": normal_consistency,
        }

    def accuracy(self, gt_points, rec_points, gt_normals=None, rec_normals=None):
        gt_points_kd_tree = KDTree(gt_points)
        distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
        acc = np.mean(distances)

        acc_median = np.median(distances)

        if gt_normals is not None and rec_normals is not None:
            normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
            normal_dot = np.abs(normal_dot)

            return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

        return acc, acc_median


    def completion(self, gt_points, rec_points, gt_normals=None, rec_normals=None):
        gt_points_kd_tree = KDTree(rec_points)
        distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
        comp = np.mean(distances)
        comp_median = np.median(distances)

        if gt_normals is not None and rec_normals is not None:
            normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
            normal_dot = np.abs(normal_dot)

            return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

        return comp, comp_median