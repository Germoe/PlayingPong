import random
from typing import TypeVar

import gymnasium as gym

Action = TypeVar("Action")
Env = TypeVar("Env")


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env: Env, epsilon=0.1):
        super().__init__(env)
        self.epsilon = epsilon

    def action(self, action: Action):
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = RandomActionWrapper(
        gym.make("CartPole-v1", render_mode="rgb_array"), epsilon=0.1
    )
    # Add monitoring of the environment to info variable
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Add video recording of the environment to "DIR"
    # Requires render_mode="rgb_array"
    # !! Requires ffmpeg (e.g. `brew install ffmpeg`)
    env = gym.wrappers.RecordVideo(env, "recording")

    obs = env.reset()
    rewards = []
    total_reward = 0
    episodes = 5

    while episodes > 0:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(0)
        total_reward += reward
        if terminated:
            rewards.append(total_reward)
            episodes -= 1
            obs = env.reset()
            total_reward = 0

    print(f"Info: {info}")

    env.close()

    print(f"Total Rewards {rewards}")
