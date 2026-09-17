"""Vision-Language Model (VLM) agent interface and response parsing."""

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VLMAction(BaseModel):
    """Structured output expected from the Vision-Language Model."""

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


class VisionLanguageAgent:
    """
    Interface for Vision-Language Models to generate motor actions from synthetic camera frames.
    """

    def __init__(
        self,
        num_dofs: int,
        mode: str = "mock",
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        self.num_dofs = num_dofs
        self.mode = mode
        self.api_key = api_key
        self.model_name = model_name
        self._step_counter = 0

    def format_system_prompt(self) -> str:
        """Constructs instructions and JSON schema constraints for the VLM."""
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

    def predict_action(self, base64_image: str, task_description: str) -> VLMAction:
        """
        Sends the captured synthetic camera frame to the VLM and parses the structured response.

        Args:
            base64_image: Base64-encoded JPEG image string.
            task_description: Natural language task specification.

        Returns:
            VLMAction: Structured action payload with joint angles and reasoning.
        """
        if self.mode == "mock":
            return self._mock_visual_policy(base64_image, task_description)
        elif self.mode == "gemini":
            return self._call_gemini_api(base64_image, task_description)
        else:
            raise NotImplementedError(f"Agent mode '{self.mode}' is not implemented.")

    def _mock_visual_policy(self, base64_image: str, task_description: str) -> VLMAction:
        """
        Simulates an intelligent multi-step visual policy reaching down towards the block.
        Progresses smoothly through visual approach phases.
        """
        self._step_counter += 1

        # Smooth staged trajectory targeting the red block
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

        # Pad with 0.0 if arm has more DOFs
        while len(angles) < self.num_dofs:
            angles.append(0.0)

        return VLMAction(
            reasoning=f"[Step {self._step_counter}] {reasoning}",
            target_joint_angles=angles,
            gripper_closed=False,
            task_completed=(self._step_counter >= len(phases)),
        )

    def _call_gemini_api(self, base64_image: str, task_description: str) -> VLMAction:
        """
        Example implementation for Google Gemini 2.5 Flash / Pro Multimodal API.
        Enforces structured JSON response validation via Pydantic.
        """
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            prompt = f"{self.format_system_prompt()}\n\nTask: {task_description}"

            image_part = types.Part.from_bytes(
                data=base64.b64decode(base64_image),
                mime_type="image/jpeg",
            )

            response = client.models.generate_content(
                model=self.model_name,
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VLMAction,
                    temperature=0.2,
                ),
            )

            parsed_data = json.loads(response.text)
            return VLMAction(**parsed_data)

        except Exception as e:
            # Fallback / graceful degradation
            raise RuntimeError(f"VLM API call failed: {e}") from e
