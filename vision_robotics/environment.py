"""Simulation environment setup for tabletop vision-guided robotics in PyBullet."""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import pybullet as p
import pybullet_data

from vision_robotics.camera import SyntheticCamera
from vision_robotics.config import SimulationConfig


class JointInfo:
    """Metadata container for a controllable robot joint."""

    def __init__(self, index: int, info: Tuple[Any, ...]):
        self.index = index
        self.name = info[1].decode("utf-8")
        self.joint_type = info[2]
        self.lower_limit = float(info[8])
        self.upper_limit = float(info[9])
        self.max_force = float(info[10])
        self.max_velocity = float(info[11])

    def clip(self, position: float) -> float:
        """Clips a target position to allowable joint limits if specified."""
        if self.lower_limit < self.upper_limit:
            return float(np.clip(position, self.lower_limit, self.upper_limit))
        return float(position)


class TabletopRobotEnv:
    """
    Simulated PyBullet tabletop robotic workspace with an overhead synthetic camera.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.client_id = p.connect(p.GUI if self.config.gui else p.DIRECT)

        # Configure search paths and global physics settings
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.config.robot.time_step)
        p.setGravity(0, 0, -9.81)

        # Camera instance
        self.camera = SyntheticCamera(self.config.camera)

        # Body identifiers
        self.plane_id: Optional[int] = None
        self.table_id: Optional[int] = None
        self.robot_id: Optional[int] = None
        self.block_id: Optional[int] = None

        # Controllable joint registry
        self.controllable_joints: List[JointInfo] = []

        # 3D In-Simulator Visual HUD tracker IDs
        self.hud_title_id: Optional[int] = None
        self.hud_sub_id: Optional[int] = None
        self.hud_target_id: Optional[int] = None
        self.beam_id: Optional[int] = None

        # Initialize the workspace
        self.reset()

    def reset(self):
        """Resets the simulation, reloads scene assets, and settles physics."""
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.config.robot.time_step)

        # 1. Ground Plane
        self.plane_id = p.loadURDF("plane.urdf")

        # 2. Table
        self.table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
        )

        # 3. Target Object (Vibrant colored cube on table surface)
        self.block_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=list(self.config.target_block_pos),
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            globalScaling=self.config.target_block_scale,
        )
        # Give the block a distinct red color for visual perception
        p.changeVisualShape(self.block_id, -1, rgbaColor=[0.9, 0.15, 0.15, 1.0])

        # 4. Robotic Arm
        self.robot_id = p.loadURDF(
            self.config.robot.urdf_path,
            basePosition=list(self.config.robot.base_position),
            baseOrientation=p.getQuaternionFromEuler(
                list(self.config.robot.base_orientation_euler)
            ),
            useFixedBase=True,
        )

        # Discover controllable joints
        self.controllable_joints.clear()
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_type = info[2]
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.controllable_joints.append(JointInfo(i, info))

        # Apply default starting pose
        default_pose = self.config.robot.default_joint_positions
        for idx, joint in enumerate(self.controllable_joints):
            pos = default_pose[idx] if idx < len(default_pose) else 0.0
            p.resetJointState(self.robot_id, joint.index, targetValue=pos)

        # Settle initial physics
        self.step_physics(self.config.robot.settle_steps, paced=False)

        # Configure 3D interactive viewport camera framing and target label in GUI mode
        if self.config.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.55,
                cameraYaw=50.0,
                cameraPitch=-32.0,
                cameraTargetPosition=[0.0, 0.05, 0.65],
            )
            self.hud_target_id = p.addUserDebugText(
                text="[TARGET CUBE]",
                textPosition=[
                    self.config.target_block_pos[0] - 0.1,
                    self.config.target_block_pos[1],
                    self.config.target_block_pos[2] + 0.14,
                ],
                textColorRGB=[1.0, 0.2, 0.2],
                textSize=1.3,
                lifeTime=0,
            )

    @property
    def num_dofs(self) -> int:
        """Returns the number of controllable degrees of freedom."""
        return len(self.controllable_joints)

    def get_joint_state_summary(self) -> List[Dict[str, Any]]:
        """Returns diagnostic joint positions for logging / telemetry."""
        states = []
        for joint in self.controllable_joints:
            pos, vel, _, _ = p.getJointState(self.robot_id, joint.index)
            states.append({
                "index": joint.index,
                "name": joint.name,
                "position": round(pos, 3),
                "velocity": round(vel, 3),
                "limits": (round(joint.lower_limit, 2), round(joint.upper_limit, 2)),
            })
        return states

    def capture_frame(self) -> Tuple[np.ndarray, Image.Image, str]:
        """Captures an RGB frame from the synthetic camera."""
        return self.camera.capture_rgb_frame()

    def step_physics(self, num_substeps: int = 1, paced: bool = True):
        """
        Advances the physics simulation for a discrete number of ticks.
        In GUI mode, pacing ensures visually natural real-time rendering.
        """
        if not p.isConnected():
            return
        dt = self.config.robot.time_step
        for _ in range(num_substeps):
            if not p.isConnected():
                break
            p.stepSimulation()
            if self.config.gui and paced:
                time.sleep(dt)

    def apply_action(
        self,
        target_joint_angles: List[float],
        substeps: Optional[int] = None,
        paced: bool = True,
    ):
        """
        Commands the robot joints to move towards the specified target angles,
        then advances the physics engine forward synchronously.

        Args:
            target_joint_angles: List of target positions for each controllable joint.
            substeps: Number of simulation steps to run during action execution.
            paced: Whether to sleep to match 240Hz physics in GUI mode.
        """
        if len(target_joint_angles) != self.num_dofs:
            raise ValueError(
                f"Action dimension mismatch: Expected {self.num_dofs} joint targets, "
                f"received {len(target_joint_angles)}."
            )

        if not p.isConnected():
            return

        # Clip commands within safe joint limits
        clipped_targets = [
            joint.clip(target)
            for joint, target in zip(self.controllable_joints, target_joint_angles)
        ]

        joint_indices = [j.index for j in self.controllable_joints]
        forces = [
            j.max_force if j.max_force > 0 else self.config.robot.motor_force
            for j in self.controllable_joints
        ]

        # Issue motor command via position control
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=clipped_targets,
            forces=forces,
        )

        # Step physics forward synchronously while robot moves to target
        steps = substeps or self.config.robot.action_substeps
        self.step_physics(steps, paced=paced)

        # Update targeting beam after joint movement
        self.update_targeting_beam()

    def update_visual_hud(
        self,
        title: str,
        subtitle: str = "",
        color: Optional[List[float]] = None,
    ):
        """Displays floating 3D HUD text directly inside the PyBullet simulation window."""
        if not self.config.gui:
            return

        main_color = color or [0.1, 1.0, 0.4]  # Bright neon green
        sub_color = [1.0, 0.9, 0.2]            # Bright gold

        text_pos_title = [-0.25, -0.30, 1.30]
        text_pos_sub = [-0.25, -0.30, 1.18]

        try:
            if self.hud_title_id is not None and self.hud_title_id >= 0:
                self.hud_title_id = p.addUserDebugText(
                    text=title,
                    textPosition=text_pos_title,
                    textColorRGB=main_color,
                    textSize=1.5,
                    lifeTime=0,
                    replaceItemUniqueId=self.hud_title_id,
                )
            else:
                self.hud_title_id = p.addUserDebugText(
                    text=title,
                    textPosition=text_pos_title,
                    textColorRGB=main_color,
                    textSize=1.5,
                    lifeTime=0,
                )

            if subtitle:
                if self.hud_sub_id is not None and self.hud_sub_id >= 0:
                    self.hud_sub_id = p.addUserDebugText(
                        text=subtitle,
                        textPosition=text_pos_sub,
                        textColorRGB=sub_color,
                        textSize=1.2,
                        lifeTime=0,
                        replaceItemUniqueId=self.hud_sub_id,
                    )
                else:
                    self.hud_sub_id = p.addUserDebugText(
                        text=subtitle,
                        textPosition=text_pos_sub,
                        textColorRGB=sub_color,
                        textSize=1.2,
                        lifeTime=0,
                    )
        except Exception:
            pass

    def update_targeting_beam(self, color: Optional[List[float]] = None):
        """Draws a 3D dynamic visual laser ray between the end-effector and target block."""
        if not self.config.gui or self.robot_id is None or self.block_id is None:
            return

        beam_color = color or [0.2, 0.8, 1.0]
        try:
            ee_pos = p.getLinkState(self.robot_id, 6)[0]
            block_pos = p.getBasePositionAndOrientation(self.block_id)[0]

            if self.beam_id is not None and self.beam_id >= 0:
                self.beam_id = p.addUserDebugLine(
                    lineFromXYZ=ee_pos,
                    lineToXYZ=block_pos,
                    lineColorRGB=beam_color,
                    lineWidth=2.5,
                    lifeTime=0,
                    replaceItemUniqueId=self.beam_id,
                )
            else:
                self.beam_id = p.addUserDebugLine(
                    lineFromXYZ=ee_pos,
                    lineToXYZ=block_pos,
                    lineColorRGB=beam_color,
                    lineWidth=2.5,
                    lifeTime=0,
                )
        except Exception:
            pass

    def close(self):
        """Disconnects the PyBullet simulation server."""
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)
