import argparse
import sys
from collections import Counter

import dqn_model as dqn_model
import numpy as np
import pong_env
import torch

EPISODES = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to model to load")
    parser.add_argument(
        "--env",
        default="ALE/Pong-v5",
        help="Name of the Pong Environment Default: ALE/Pong-v5",
    )
    parser.add_argument(
        "-r",
        "--record",
        default=False,
        action="store_true",
        help="Record a video of the agent playing",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        default=EPISODES,
        help="Number of episodes to run the agent for",
    )

    args = parser.parse_args()

    env = pong_env.make_env(args.env, args.record, 1)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))

    episodes = int(args.episodes)
    rewards = []

    for i in range(episodes):
        total_reward = 0.0

        obs, _ = env.reset()
        c: Counter = Counter()

        while True:
            # Select Action
            obs_a = np.array([obs])

            print(np.mean(obs_a), np.std(obs_a))
            obs_v = torch.tensor(obs_a, dtype=torch.float32)
            q_vals_v = net(obs_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = act_v.item()
            c[action] += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

            obs = next_obs

        rewards.append(total_reward)

        sys.stdout.write(
            f"\rAverage Reward: {np.mean(rewards):.4f} for {i} episodes in {args.env}"
        )
        sys.stdout.flush()

    env.close()
