import argparse

import cv2
import gymnasium as gym


def grayscale_observation(obs):
    """
    Grayscale observation following the human eye's perception of color
    """
    return obs[:, :, 0] * 0.299 + obs[:, :, 1] * 0.587 + obs[:, :, 2] * 0.114


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ALE/Pong-v5")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    env_name = args.env
    steps = args.steps

    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()

    cv2.imwrite(
        "screenshots/pong_{}.png".format(0), cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    )

    # Play n steps
    for i in range(1, steps + 1):
        action = env.action_space.sample()
        next_obs, _, _, _, _ = env.step(action)
        # Grayscale Observation Human Eye
        cv2.imwrite(
            "screenshots/pong_{}.png".format(i),
            cv2.cvtColor(next_obs, cv2.COLOR_BGR2RGB),
        )
