#!/usr/bin/env python3
"""
Example training loop using the AIC Gym environment.

This demonstrates how to use the environment for RL training.
Replace the random policy with your own policy implementation.

Prerequisites:
  1. Simulation running (via /entrypoint.sh in distrobox)
  2. Resetter node running (via reset_env.sh in distrobox)

Usage:
  pixi run python aic_utils/aic_training/example_training_loop.py
"""

import numpy as np
from aic_training.aic_gym_env import AICGymEnv, AICGymEnvConfig


def main():
    # Configure environment
    config = AICGymEnvConfig(
        control_mode="cartesian",
        control_freq_hz=10.0,
        max_episode_steps=500,  # 50 seconds at 10 Hz
    )

    env = AICGymEnv(config=config)

    num_episodes = 10
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")

        obs, info = env.reset()
        print(f"Initial obs shape: {obs.shape}")
        print(f"TCP position: ({obs[0]:.4f}, {obs[1]:.4f}, {obs[2]:.4f})")

        episode_reward = 0.0
        step = 0

        while True:
            # --- Replace this with your policy ---
            action = env.action_space.sample() * 0.1  # small random actions
            # --- End policy ---

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

            if step % 50 == 0:
                print(
                    f"  Step {step}: "
                    f"TCP=({obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}), "
                    f"reward={reward:.3f}, "
                    f"insertion={info['insertion_event']}"
                )

            if terminated:
                print(f"  Episode terminated (insertion success!)")
                break
            if truncated:
                print(f"  Episode truncated (max steps reached)")
                break

        print(f"Episode {episode + 1} finished: {step} steps, reward={episode_reward:.3f}")

    env.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
