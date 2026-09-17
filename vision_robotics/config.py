"""Configuration settings for the Vision Robotics Simulation."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CameraConfig:
    """Virtual synthetic camera configuration."""
    width: int = 640
    height: int = 480
    target_position: Tuple[float, float, float] = (0.0, 0.0, 0.65)  # Workspace table focus
    distance: float = 1.4                                          # Distance from camera target
    yaw: float = 50.0                                              # Angle around z-axis in degrees
    pitch: float = -35.0                                           # Elevation angle in degrees
    roll: float = 0.0
    fov: float = 60.0                                              # Field of view in degrees
    near_val: float = 0.1
    far_val: float = 3.5


@dataclass
class RobotConfig:
    """Robotic arm and workspace physics configuration."""
    urdf_path: str = "kuka_iiwa/model.urdf"
    base_position: Tuple[float, float, float] = (-0.4, 0.0, 0.625)
    base_orientation_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_joint_positions: List[float] = field(
        default_factory=lambda: [0.0, 0.4, 0.0, -1.2, 0.0, 0.8, 0.0]
    )
    motor_force: float = 200.0
    settle_steps: int = 60
    action_substeps: int = 120  # Physics steps per high-level action step
    time_step: float = 1.0 / 240.0


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""
    gui: bool = True
    camera: CameraConfig = field(default_factory=CameraConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    target_block_pos: Tuple[float, float, float] = (0.25, 0.15, 0.65)
    target_block_scale: float = 1.5
