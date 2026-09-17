"""Automated test suite for Vision-Driven Robotics simulation."""

import base64
from io import BytesIO
import unittest
import numpy as np
from PIL import Image

from vision_robotics import (
    CameraConfig,
    RobotConfig,
    SimulationConfig,
    TabletopRobotEnv,
    VisionLanguageAgent,
    VLMAction,
)


class TestVisionRoboticsEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SimulationConfig(
            gui=False,  # Run headless for CI/test speed
            camera=CameraConfig(width=320, height=240),
            robot=RobotConfig(action_substeps=30, settle_steps=10),
        )
        cls.env = TabletopRobotEnv(config=cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_controllable_joints(self):
        """Verify arm has controllable degrees of freedom."""
        self.assertGreater(self.env.num_dofs, 0)
        self.assertEqual(self.env.num_dofs, 7)

    def test_camera_frame_capture(self):
        """Verify RGB camera buffer dimensions and base64 formatting."""
        rgb_arr, pil_img, b64_str = self.env.capture_frame()

        self.assertIsInstance(rgb_arr, np.ndarray)
        self.assertEqual(rgb_arr.shape, (240, 320, 3))
        self.assertEqual(rgb_arr.dtype, np.uint8)

        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.size, (320, 240))

        # Check valid base64
        decoded = base64.b64decode(b64_str)
        self.assertGreater(len(decoded), 1000)
        reopened = Image.open(BytesIO(decoded))
        self.assertEqual(reopened.size, (320, 240))

    def test_agent_mock_policy(self):
        """Verify VLM agent generates valid structured motor actions."""
        agent = VisionLanguageAgent(num_dofs=self.env.num_dofs, mode="mock")
        _, _, b64_str = self.env.capture_frame()

        action = agent.predict_action(b64_str, "Reach for the red target block")
        self.assertIsInstance(action, VLMAction)
        self.assertEqual(len(action.target_joint_angles), self.env.num_dofs)
        self.assertTrue(len(action.reasoning) > 0)

    def test_action_execution_and_physics_step(self):
        """Verify physics engine executes joint commands without errors."""
        target_angles = [0.2, 0.4, 0.0, -1.0, 0.0, 0.5, 0.0]
        self.env.apply_action(target_angles, substeps=10, paced=False)

        states = self.env.get_joint_state_summary()
        self.assertEqual(len(states), self.env.num_dofs)

    def test_jev_system_one_live(self):
        """Verify live Jev System One model integration if TYPESAFE_API_KEY is configured."""
        import os
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            self.skipTest("TYPESAFE_API_KEY not found in environment.")

        agent = VisionLanguageAgent(num_dofs=self.env.num_dofs, mode="jev")
        _, _, b64_str = self.env.capture_frame()
        action = agent.predict_action(
            base64_image=b64_str,
            task_description="Reach for the red target cube",
            joint_summary=self.env.get_joint_state_summary(),
        )
        self.assertIsNotNone(action.jev_data)
        self.assertIn("jev", action.jev_data["model"])
        self.assertIn("action_phase", action.jev_data["full_choices"])
        self.assertIn("alignment_confidence", action.jev_data["full_nouls"])


if __name__ == "__main__":
    unittest.main()
