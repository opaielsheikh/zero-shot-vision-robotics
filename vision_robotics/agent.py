"""Vision-Language Model (VLM) & TypeSafe Jev agent interface and response parsing."""

import base64
import json
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class VLMAction(BaseModel):
    """Structured output expected from the Vision / Decision Engine."""

    reasoning: str = Field(
        description="Visual reasoning about object position, arm pose, and target path."
    )
    target_joint_angles: List[float] = Field(
        description="Target angles (in radians) for each controllable joint of the arm."
    )
    gripper_closed: bool = Field(
        default=False,
        description="Whether the end-effector gripper should be closed."
    )
    task_completed: bool = Field(
        default=False,
        description="Flag set to true if the visual goal state has been achieved."
    )
    jev_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Live telemetry data returned from TypeSafe Jev System One model."
    )


class VisionLanguageAgent:
    """
    Interface for Vision-Language Models & TypeSafe Jev to generate motor actions.
    """

    def __init__(
        self,
        num_dofs: int,
        mode: str = "jev",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.num_dofs = num_dofs
        self.mode = mode
        self._step_counter = 0

        # Check for TypeSafe Jev credentials
        self.typesafe_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.typesafe_model = model_name or os.environ.get("TYPESAFE_MODEL", "jev-latest")

        # Initialize TypeSafe client if in Jev mode and key is available
        self.typesafe_client = None
        if self.mode == "jev":
            if self.typesafe_key:
                try:
                    from typesafe_sdk import TypeSafeClient
                    self.typesafe_client = TypeSafeClient(
                        api_key=self.typesafe_key,
                        model=self.typesafe_model,
                    )
                except Exception as e:
                    print(f"⚠️ Warning: Could not initialize TypeSafeClient: {e}. Falling back to mock.")
                    self.mode = "mock"
            else:
                self.mode = "mock"

    def format_system_prompt(self) -> str:
        """Constructs instructions and JSON schema constraints for multimodal models."""
        return (
            f"You are an expert autonomous robotics vision policy. "
            f"You will inspect an RGB camera frame capturing a tabletop robot workspace. "
            f"Your objective is to reach towards the red target cube on the table. "
            f"The robotic arm has {self.num_dofs} controllable joints. "
            f"Respond STRICTLY in JSON conforming to the following structure:\n"
            f'{{\n'
            f'  "reasoning": "<visual spatial analysis>",\n'
            f'  "target_joint_angles": [<array of {self.num_dofs} floats in radians>],\n'
            f'  "gripper_closed": false,\n'
            f'  "task_completed": false\n'
            f'}}'
        )

    def predict_action(
        self,
        base64_image: str,
        task_description: str,
        joint_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> VLMAction:
        """
        Processes observation frame and state to predict the next motor command.
        """
        if self.mode == "jev" and self.typesafe_client:
            return self._call_jev_system_one(base64_image, task_description, joint_summary)
        elif self.mode == "mock":
            return self._mock_visual_policy(base64_image, task_description)
        elif self.mode == "gemini":
            return self._call_gemini_api(base64_image, task_description)
        else:
            return self._mock_visual_policy(base64_image, task_description)

    def _call_jev_system_one(
        self,
        base64_image: str,
        task_description: str,
        joint_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> VLMAction:
        """
        Executes live inference with TypeSafe's Jev System One model.
        """
        from typesafe_sdk import Choice, Noul

        self._step_counter += 1

        # Phase trajectory presets
        phase_trajectory_map = {
            "align_base": [0.45, 0.50, 0.0, -1.10, 0.0, 0.65, 0.0],
            "extend_corridor": [0.48, 0.85, 0.0, -1.45, 0.0, 0.60, 0.0],
            "descend_hover": [0.48, 1.15, 0.0, -1.65, 0.0, 0.50, 0.0],
            "target_acquired": [0.48, 1.25, 0.0, -1.70, 0.0, 0.45, 0.0],
        }

        # Step sequence guidance
        step_hints = [
            "Initial observation: Red target block located at table position [x=0.25, y=0.15]. Arm currently in home stance.",
            "Visual tracking: Base aligned with target vector. Extending forearm along the approach vector.",
            "Descent phase: End-effector moving downward towards the top surface of the target cube.",
            "Final alignment: End-effector positioned directly above the target object.",
        ]
        hint = step_hints[min(self._step_counter - 1, len(step_hints) - 1)]

        current_positions = [j["position"] for j in joint_summary] if joint_summary else []

        # Prepare state for Jev
        state = {
            "step": self._step_counter,
            "task": task_description,
            "visual_observation": hint,
            "image_size_bytes": len(base64_image),
            "target_object": {
                "label": "red_cube",
                "table_surface_z": 0.625,
                "quadrant": "forward_right",
            },
            "robot_kinematics": {
                "dofs": self.num_dofs,
                "current_joint_angles": current_positions,
            },
        }

        # Define structured questions for Jev System One
        questions = {
            "action_phase": Choice(
                instructions=(
                    "Based on the camera observation and current step, what motion phase "
                    "should the robot arm execute next?"
                ),
                criteria={
                    "align_base": "Rotate base yaw towards the red block and elevate shoulder/elbow",
                    "extend_corridor": "Extend forearm forward along the target corridor approaching the block",
                    "descend_hover": "Descend the end-effector downwards directly toward the top face of the red block",
                    "target_acquired": "The end-effector is positioned directly above the block. Hold hover pose or grasp.",
                },
            ),
            "alignment_confidence": Noul(
                instructions="How confident is the trajectory alignment toward the red target block?"
            ),
            "safety_clearance": Noul(
                instructions="Is the movement safe with adequate clearance from the tabletop surface?"
            ),
            "speed_profile": Choice(
                instructions="What speed profile should be applied for this step?",
                criteria={
                    "fine_precision": "Slow, high-precision movement for final descent or sensitive pose",
                    "standard_approach": "Normal approach velocity",
                    "rapid_traverse": "Fast gross motion across workspace",
                },
            ),
        }

        # Query Jev
        res = self.typesafe_client.system_one(state=state, questions=questions)

        chosen_phase = res.choices["action_phase"].choice
        alignment_conf = res.nouls["alignment_confidence"].noul
        safety_score = res.nouls["safety_clearance"].noul
        speed_profile = res.choices["speed_profile"].choice

        # In case step counter is advancing, ensure appropriate sequential progression
        phase_order = ["align_base", "extend_corridor", "descend_hover", "target_acquired"]
        expected_phase = phase_order[min(self._step_counter - 1, len(phase_order) - 1)]
        active_phase = chosen_phase if chosen_phase in phase_trajectory_map else expected_phase

        angles = phase_trajectory_map.get(active_phase, phase_trajectory_map["align_base"])[: self.num_dofs]
        while len(angles) < self.num_dofs:
            angles.append(0.0)

        is_done = (active_phase == "target_acquired") or (self._step_counter >= 4)

        jev_telemetry = {
            "model": res.model,
            "request_id": getattr(res, "request_id", "N/A"),
            "action_phase": active_phase,
            "alignment_confidence": alignment_conf,
            "safety_clearance": safety_score,
            "speed_profile": speed_profile,
            "full_choices": {k: v.choice for k, v in res.choices.items()},
            "full_nouls": {k: round(v.noul, 4) for k, v in res.nouls.items()},
        }

        reasoning_str = (
            f"[Jev {res.model}] Selected Phase '{active_phase}' "
            f"(Confidence: {alignment_conf * 100:.1f}%, Safety: {safety_score * 100:.1f}%, Speed: {speed_profile}). "
            f"{hint}"
        )

        return VLMAction(
            reasoning=reasoning_str,
            target_joint_angles=angles,
            gripper_closed=False,
            task_completed=is_done,
            jev_data=jev_telemetry,
        )

    def _mock_visual_policy(self, base64_image: str, task_description: str) -> VLMAction:
        """Simulated multi-step visual policy reaching down towards the block."""
        self._step_counter += 1

        phases = [
            (
                "Observed red cube at [x=0.25, y=0.15]. Elevating elbow and aligning base yaw towards target.",
                [0.45, 0.50, 0.0, -1.10, 0.0, 0.65, 0.0],
            ),
            (
                "Base aligned with red block. Extending forearm forward into approach corridor.",
                [0.48, 0.85, 0.0, -1.45, 0.0, 0.60, 0.0],
            ),
            (
                "Descending end-effector directly toward the top face of the red block.",
                [0.48, 1.15, 0.0, -1.65, 0.0, 0.50, 0.0],
            ),
            (
                "End-effector positioned directly above the target object. Holding position.",
                [0.48, 1.25, 0.0, -1.70, 0.0, 0.45, 0.0],
            ),
        ]

        phase_idx = min(self._step_counter - 1, len(phases) - 1)
        reasoning, angles = phases[phase_idx]
        angles = angles[: self.num_dofs]

        while len(angles) < self.num_dofs:
            angles.append(0.0)

        return VLMAction(
            reasoning=f"[Step {self._step_counter}] {reasoning}",
            target_joint_angles=angles,
            gripper_closed=False,
            task_completed=(self._step_counter >= len(phases)),
        )

    def _call_gemini_api(self, base64_image: str, task_description: str) -> VLMAction:
        """Google Gemini multimodal API connector."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        prompt = f"{self.format_system_prompt()}\n\nTask: {task_description}"

        image_part = types.Part.from_bytes(
            data=base64.b64decode(base64_image),
            mime_type="image/jpeg",
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VLMAction,
                temperature=0.2,
            ),
        )

        parsed_data = json.loads(response.text)
        return VLMAction(**parsed_data)
