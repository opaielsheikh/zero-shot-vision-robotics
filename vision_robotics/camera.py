"""Virtual synthetic camera module for PyBullet simulations."""

import base64
from io import BytesIO
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import pybullet as p

from vision_robotics.config import CameraConfig


class SyntheticCamera:
    """Controls a virtual RGB-D synthetic camera positioned in a PyBullet simulation."""

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self._precompute_matrices()

    def _precompute_matrices(self):
        """Calculates projection matrix and initial view matrix."""
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.fov,
            aspect=float(self.config.width) / float(self.config.height),
            nearVal=self.config.near_val,
            farVal=self.config.far_val,
        )

    def get_view_matrix(self) -> Tuple[float, ...]:
        """Computes view matrix from current extrinsic camera pose settings."""
        return p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=list(self.config.target_position),
            distance=self.config.distance,
            yaw=self.config.yaw,
            pitch=self.config.pitch,
            roll=self.config.roll,
            upAxisIndex=2,  # Z is UP
        )

    def capture_rgb_frame(self) -> Tuple[np.ndarray, Image.Image, str]:
        """
        Captures an RGB snapshot from the synthetic camera.

        Returns:
            rgb_array: Raw uint8 numpy array of shape (height, width, 3).
            pil_image: PIL Image object suitable for visual debugging or saving.
            base64_str: Base64-encoded JPEG image string ready for VLM JSON API payloads.
        """
        view_matrix = self.get_view_matrix()

        # Choose hardware renderer if GUI is connected, else CPU tiny renderer
        renderer = (
            p.ER_BULLET_HARDWARE_OPENGL
            if p.getConnectionInfo()["isConnected"] and p.getConnectionInfo()["connectionMethod"] == p.GUI
            else p.ER_TINY_RENDERER
        )

        # Capture virtual camera buffer
        _, _, rgb_raw, _, _ = p.getCameraImage(
            width=self.config.width,
            height=self.config.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=renderer,
            flags=p.ER_NO_SEGMENTATION_MASK,
        )

        # PyBullet outputs flat or (H, W, 4) RGBA buffer
        rgba_array = np.reshape(
            rgb_raw, (self.config.height, self.config.width, 4)
        ).astype(np.uint8)

        # Extract RGB channels
        rgb_array = rgba_array[:, :, :3]

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_array)

        # Convert to Base64 JPEG
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return rgb_array, pil_image, base64_str

    def capture_depth_map(self) -> np.ndarray:
        """
        Captures linear depth values in meters.

        Returns:
            depth_meters: float32 numpy array of shape (height, width).
        """
        view_matrix = self.get_view_matrix()
        renderer = (
            p.ER_BULLET_HARDWARE_OPENGL
            if p.getConnectionInfo()["isConnected"] and p.getConnectionInfo()["connectionMethod"] == p.GUI
            else p.ER_TINY_RENDERER
        )
        _, _, _, depth_raw, _ = p.getCameraImage(
            width=self.config.width,
            height=self.config.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=renderer,
            flags=p.ER_NO_SEGMENTATION_MASK,
        )

        depth_buf = np.reshape(depth_raw, (self.config.height, self.config.width))
        near = self.config.near_val
        far = self.config.far_val
        # Linearize OpenGL depth buffer
        depth_meters = far * near / (far - (far - near) * depth_buf)
        return depth_meters.astype(np.float32)
