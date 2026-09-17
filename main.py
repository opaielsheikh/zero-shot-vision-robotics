"""Main entrypoint for the Vision-Driven Robotics Simulation Demo."""

import argparse
import json
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

from vision_robotics import (
    CameraConfig,
    RobotConfig,
    SimulationConfig,
    TabletopRobotEnv,
    VisionLanguageAgent,
)


def print_jev_banner(jev_data: dict, step: int, max_steps: int):
    """Renders a formatted telemetry panel for the Jev model output."""
    model = jev_data.get("model", "jev-latest")
    phase = jev_data.get("action_phase", "unknown")
    conf = jev_data.get("alignment_confidence", 0.0)
    safety = jev_data.get("safety_clearance", 0.0)
    speed = jev_data.get("speed_profile", "standard")
    req_id = jev_data.get("request_id", "N/A")

    print("\n" + "┌" + "─" * 68 + "┐")
    print(f"│ 🧠 TYPE-SAFE JEV MODEL INFERENCE [{model}]".ljust(69) + "│")
    print(f"│ Request ID: {req_id}".ljust(69) + "│")
    print("├" + "─" * 68 + "┤")
    print(f"│ 🎯 Decision Phase    : {phase.upper()}".ljust(69) + "│")
    print(f"│ 📊 Alignment Conf.   : {conf * 100:.1f}% (Noul Score)".ljust(69) + "│")
    print(f"│ 🛡️ Safety Clearance  : {safety * 100:.1f}% (Noul Score)".ljust(69) + "│")
    print(f"│ ⚡ Speed Profile     : {speed}".ljust(69) + "│")
    print("├" + "─" * 68 + "┤")
    print("│ Raw Model Output:".ljust(69) + "│")
    raw_choices = json.dumps(jev_data.get("full_choices", {}))
    raw_nouls = json.dumps(jev_data.get("full_nouls", {}))
    print(f"│   Choices: {raw_choices}".ljust(69) + "│")
    print(f"│   Nouls  : {raw_nouls}".ljust(69) + "│")
    print("└" + "─" * 68 + "┘\n")


def run_vision_robotics_demo(
    headless: bool = False,
    max_steps: int = 4,
    save_frames: bool = True,
    output_dir: str = "captured_frames",
):
    """
    Executes the turn-based vision-driven perception-action loop with live
    inference from TypeSafe Jev System One model.
    """
    if save_frames:
        os.makedirs(output_dir, exist_ok=True)

    # 1. Simulation Configuration & Environment Setup
    config = SimulationConfig(
        gui=not headless,
        camera=CameraConfig(width=640, height=480, fov=60.0),
        robot=RobotConfig(action_substeps=120),  # 120 ticks @ 240Hz = 0.5s execution window
    )

    print("=" * 70)
    print("🚀 Initializing Vision-Driven Robotics Environment (PyBullet)")
    print("=" * 70)

    env = TabletopRobotEnv(config=config)
    print(f"✅ Environment initialized.")
    print(f"🤖 Controllable Arm DOFs: {env.num_dofs}")
    print(f"📷 Synthetic Camera: {config.camera.width}x{config.camera.height} FOV={config.camera.fov}°")
    print(f"🎯 Target: Red cube positioned on the tabletop.")

    # 2. Check for TypeSafe Jev model credentials
    typesafe_key = os.environ.get("TYPESAFE_API_KEY")
    typesafe_model = os.environ.get("TYPESAFE_MODEL", "jev-latest")

    if typesafe_key:
        agent_mode = "jev"
        print(f"🧠 TypeSafe Jev Agent: ENABLED (Model: {typesafe_model})")
    else:
        agent_mode = "mock"
        print("ℹ️ No TYPESAFE_API_KEY found in environment. Using simulated visual policy.")

    print("=" * 70)

    agent = VisionLanguageAgent(
        num_dofs=env.num_dofs,
        mode=agent_mode,
        api_key=typesafe_key,
        model_name=typesafe_model,
    )
    task_prompt = "Locate the red cube on the tabletop and align the robotic arm to reach above it."

    # Initial 3D HUD banner in simulation window
    env.update_visual_hud(
        title=f"JEV SYSTEM ONE [{typesafe_model}]",
        subtitle="Perceiving workspace & red target cube...",
    )
    env.update_targeting_beam()

    try:
        for step in range(1, max_steps + 1):
            print(f"\n==================== [PERCEPTION-ACTION STEP {step}/{max_steps}] ====================")

            # Update HUD status in the 3D window
            env.update_visual_hud(
                title=f"STEP {step}/{max_steps}: SYNTHETIC CAMERA CAPTURE",
                subtitle="Transmitting RGB frame to Jev model...",
            )

            # -------------------------------------------------------------
            # STEP A: PAUSE SIMULATION & CAPTURE SYNTHETIC CAMERA RGB FRAME
            # -------------------------------------------------------------
            t0 = time.time()
            rgb_arr, pil_img, b64_img = env.capture_frame()
            capture_time_ms = (time.time() - t0) * 1000

            print(f"📸 Frame Captured: shape={rgb_arr.shape}, base64={len(b64_img)} bytes ({capture_time_ms:.1f}ms)")

            if save_frames:
                frame_path = os.path.join(output_dir, f"step_{step:02d}.jpg")
                pil_img.save(frame_path)
                print(f"💾 Saved camera snapshot to: {frame_path}")

            # -------------------------------------------------------------
            # STEP B: QUERY JEV / VISION MODEL
            # -------------------------------------------------------------
            print(f"🧠 Querying {agent.typesafe_model if agent_mode == 'jev' else 'Visual Agent'}...")
            t_model = time.time()
            current_joint_states = env.get_joint_state_summary()
            action = agent.predict_action(
                base64_image=b64_img,
                task_description=task_prompt,
                joint_summary=current_joint_states,
            )
            model_time_ms = (time.time() - t_model) * 1000

            # Display Jev System One Telemetry in Terminal AND in 3D PyBullet GUI
            if action.jev_data:
                print_jev_banner(action.jev_data, step, max_steps)
                phase_name = action.jev_data.get("action_phase", "").upper()
                model_name = action.jev_data.get("model", "jev-latest")
                conf_val = action.jev_data.get("alignment_confidence", 0.0) * 100
                safe_val = action.jev_data.get("safety_clearance", 0.0) * 100
                speed_val = action.jev_data.get("speed_profile", "standard")

                env.update_visual_hud(
                    title=f"JEV [{model_name}]: {phase_name}",
                    subtitle=f"Conf: {conf_val:.0f}% | Safety: {safe_val:.0f}% | Speed: {speed_val}",
                )
            else:
                print(f"💡 Agent Reasoning: {action.reasoning}")
                env.update_visual_hud(
                    title=f"ACTION STEP {step}",
                    subtitle=action.reasoning[:45],
                )

            env.update_targeting_beam()

            formatted_angles = [round(a, 3) for a in action.target_joint_angles]
            print(f"🎯 Dispatched Motor Target Angles (rad): {formatted_angles} ({model_time_ms:.1f}ms)")

            # -------------------------------------------------------------
            # STEP C: EXECUTE ACTION & STEP PHYSICS FORWARD
            # -------------------------------------------------------------
            print(f"⚙️ Advancing PyBullet physics ({config.robot.action_substeps} ticks @ 240Hz)...")
            env.apply_action(
                target_joint_angles=action.target_joint_angles,
                substeps=config.robot.action_substeps,
                paced=not headless,
            )

            # Telemetry verification
            active_joint = env.get_joint_state_summary()[1]  # Shoulder joint
            print(f"📊 Joint 1 State Post-Execution: pos={active_joint['position']} rad")

            if action.task_completed:
                env.update_visual_hud(
                    title="🎯 TARGET ACQUIRED (GOAL REACHED)",
                    subtitle="Task Completed Successfully by Jev Model!",
                    color=[0.0, 1.0, 0.2],
                )
                print("\n🎉 Target acquired! Goal configuration reached.")
                break

        print("\n" + "=" * 70)
        print("🏁 Simulation loop completed successfully.")
        print("=" * 70)

        if not headless:
            print("Holding final scene for 2 seconds...")
            time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n⚠️ Simulation interrupted by user.")
    finally:
        env.close()
        print("🔌 Simulation server disconnected.")


def main():
    parser = argparse.ArgumentParser(description="PyBullet Vision-Language Robotics Simulation")
    parser.add_argument("--headless", action="store_true", help="Run without PyBullet GUI")
    parser.add_argument("--steps", type=int, default=4, help="Number of decision steps")
    parser.add_argument("--output-dir", type=str, default="captured_frames", help="Directory to save frames")
    args = parser.parse_args()

    run_vision_robotics_demo(
        headless=args.headless,
        max_steps=args.steps,
        save_frames=True,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
