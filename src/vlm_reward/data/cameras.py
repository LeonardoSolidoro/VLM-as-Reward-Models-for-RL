"""
Camera policies shared by the progress and reward-data generators.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sapien
from mani_skill.utils import sapien_utils
from scipy.spatial.transform import Rotation

from .common import CAMERA_MODES


@dataclass(frozen=True)
class MovingCameraParameters:
    """
    Trajectory-level parameters for the free-moving camera.
    """
    base_azimuth_degrees: float
    base_elevation_degrees: float
    base_radius: float

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "MovingCameraParameters":
        return cls(
            base_azimuth_degrees=float(rng.uniform(-150.0, -90.0)),
            base_elevation_degrees=float(rng.uniform(45.0, 60.0)),
            base_radius=float(rng.uniform(0.9, 1.1)),
        )


def validate_camera_mode(camera_mode: str) -> str:
    if camera_mode not in CAMERA_MODES:
        raise ValueError(f"Unsupported camera mode {camera_mode!r}; choose from {CAMERA_MODES}")
    return camera_mode


def camera_pose_from_spherical(
    target: np.ndarray,
    azimuth_degrees: float,
    elevation_degrees: float,
    radius: float,
) -> sapien.Pose:
    if target.shape != (3,):
        raise ValueError(f"Camera target must have shape (3,), got {target.shape}")
    if radius <= 0.0:
        raise ValueError(f"Camera radius must be positive, got {radius}")

    azimuth = np.deg2rad(azimuth_degrees)
    elevation = np.deg2rad(elevation_degrees)
    position = target + radius * np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float64,
    )

    forward = target - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm <= 1e-12:
        raise ValueError("Camera position and target must differ")
    forward /= forward_norm

    world_up = np.array([0.0, 0.0, 1.0])
    left = np.cross(world_up, forward)
    left_norm = np.linalg.norm(left)
    if left_norm <= 1e-12:
        raise ValueError("Camera forward direction cannot be parallel to world up")
    left /= left_norm
    up = np.cross(forward, left)

    rotation_matrix = np.stack([forward, left, up], axis=1)
    quaternion_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    quaternion_wxyz = np.array(
        [
            quaternion_xyzw[3],
            quaternion_xyzw[0],
            quaternion_xyzw[1],
            quaternion_xyzw[2],
        ]
    )
    return sapien.Pose(p=position, q=quaternion_wxyz)


def task_camera_target(env: Any, task: str) -> np.ndarray:
    if task == "PickCube-v1":
        object_position = env.cube.pose.p[0].cpu().numpy()
        goal_position = env.goal_site.pose.p[0].cpu().numpy()
        return 0.6 * object_position + 0.4 * goal_position
    if task == "PushCube-v1":
        object_position = env.obj.pose.p[0].cpu().numpy()
        goal_position = env.goal_region.pose.p[0].cpu().numpy()
        return 0.5 * object_position + 0.5 * goal_position
    if task == "PegInsertionSide-v1":
        object_position = env.peg.pose.p[0].cpu().numpy()
        goal_position = env.goal_pose.p[0].cpu().numpy()
        return 0.5 * object_position + 0.5 * goal_position
    raise ValueError(f"Unsupported task for wrist-mounted camera: {task}")


def update_moving_camera(
    env: Any,
    frame_index: int,
    num_frames: int,
    parameters: MovingCameraParameters,
    rng: np.random.Generator,
) -> None:
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if frame_index < 0 or frame_index >= num_frames:
        raise ValueError(f"Frame index {frame_index} is outside [0, {num_frames})")

    time = frame_index / max(num_frames - 1, 1)
    azimuth = (
        parameters.base_azimuth_degrees
        + 10.0 * np.sin(np.pi * time)
        + rng.uniform(-0.5, 0.5)
    )
    elevation = (
        parameters.base_elevation_degrees
        + 10.0 * np.sin(2.0 * np.pi * time)
        + rng.uniform(-0.5, 0.5)
    )
    radius = parameters.base_radius + 0.02 * np.sin(np.pi * time)
    target = np.array(
        [
            0.10 * np.sin(2.0 * np.pi * time) + rng.uniform(-0.005, 0.005),
            0.10 * np.cos(2.6 * np.pi * time) + rng.uniform(-0.005, 0.005),
            0.2 + rng.uniform(-0.005, 0.005),
        ]
    )

    camera = env.scene.human_render_cameras["render_camera"].camera
    camera.set_local_pose(camera_pose_from_spherical(target, azimuth, elevation, radius))


def update_wrist_mounted_camera(env: Any, task: str) -> None:
    wrist_link = env.agent.robot.links_map["panda_hand"]
    wrist_position = wrist_link.pose.p[0].cpu().numpy()
    offset = (
        np.array([0.10, -0.10, 0.28])
        if task == "PickCube-v1"
        else np.array([0.065, -0.065, 0.25])
    )
    eye = wrist_position + offset
    target = task_camera_target(env, task)
    pose = sapien_utils.look_at(eye=eye, target=target)
    camera = env.scene.human_render_cameras["render_camera"].camera
    camera.set_local_pose(pose.sp)


def configure_camera(
    env: Any,
    task: str,
    camera_mode: str,
    frame_index: int,
    num_frames: int,
    moving_parameters: MovingCameraParameters | None,
    rng: np.random.Generator,
) -> None:
    """
    Apply the requested camera policy to the environment's render camera.
    """
    validate_camera_mode(camera_mode)
    if camera_mode == "static":
        if moving_parameters is not None:
            raise ValueError("Static camera mode must not receive moving-camera parameters")
        return
    if camera_mode == "moving":
        if moving_parameters is None:
            raise ValueError("Moving camera mode requires trajectory-level parameters")
        update_moving_camera(env, frame_index, num_frames, moving_parameters, rng)
        return
    if moving_parameters is not None:
        raise ValueError("Wrist-mounted camera mode must not receive moving-camera parameters")
    update_wrist_mounted_camera(env, task)
