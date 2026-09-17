"""Main entrypoint for the Vision-Driven Robotics Simulation Demo."""

import argparse
import os
import sys
import time

from vision_robotics import (
    CameraConfig,
    RobotConfig,
    SimulationConfig,
    TabletopRobotEnv,
    VisionLanguageAgent,
)


def run_vision_robotics_demo(
    headless: bool = False,
    max_steps: int = 5,
    save_frames: bool = True,
    output_dir: str = "captured_frames",
):
    """
    Executes the turn-based vision-driven perception-action loop.
    Physics pauses while the VLM perceives the scene and deliberates,
    eliminating network latency issues.
    """
    if save_frames:
        os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize Simulation Configuration & Environment
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
    print("=" * 70)

    # 2. Initialize VLM Agent
    agent = VisionLanguageAgent(num_dofs=env.num_dofs, mode="mock")
    task_prompt = "Locate the red cube on the tabletop and align the robotic arm to reach above it."

    try:
        for step in range(1, max_steps + 1):
            print(f"\n--- [PERCEPTION-ACTION STEP {step}/{max_steps}] ---")

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
            # STEP B: QUERY VISION-LANGUAGE MODEL (VLM)
            # -------------------------------------------------------------
            print(f"🧠 Querying VLM with visual frame & task: '{task_prompt}'...")
            t_model = time.time()
            action = agent.predict_action(
                base64_image=b64_img,
                task_description=task_prompt,
            )
            model_time_ms = (time.time() - t_model) * 1000

            print(f"💡 VLM Reasoning: {action.reasoning}")
            formatted_angles = [round(a, 3) for a in action.target_joint_angles]
            print(f"🎯 Target Joint Angles (rad): {formatted_angles} ({model_time_ms:.1f}ms)")

            # -------------------------------------------------------------
            # STEP C: EXECUTE ACTION & STEP PHYSICS FORWARD
            # -------------------------------------------------------------
            print(f"⚙️ Applying joint position control & stepping physics ({config.robot.action_substeps} ticks)...")
            env.apply_action(
                target_joint_angles=action.target_joint_angles,
                substeps=config.robot.action_substeps,
                paced=not headless,
            )

            # Telemetry verification
            active_joint = env.get_joint_state_summary()[1]  # Shoulder / elbow
            print(f"📊 Current Joint 1 State: pos={active_joint['position']} rad")

            if action.task_completed:
                print("\n🎉 VLM indicates visual task goal has been reached!")
                break

        print("\n" + "=" * 70)
        print("🏁 Simulation loop completed successfully.")
        print("=" * 70)

        # Allow user to inspect final pose in GUI mode
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
