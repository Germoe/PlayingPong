from collections import namedtuple

import gymnasium as gym
import gymnasium.wrappers as wrappers
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


Episode = namedtuple("Episode", ["reward", "steps"])
EpisodeStep = namedtuple("EpisodeStep", ["observation", "action"])


class Agent(nn.Module):
    def __init__(self, obs_size, hidden_size, action_size):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return self.pipe(x)


def iterate_batches(env, agent, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.tensor([obs])
        action_probs_v = sm(agent(obs_v))
        action_probs = action_probs_v.data.numpy()[
            0
        ]  # Select the first batch from the neural net
        action = np.random.choice(
            len(action_probs), p=action_probs
        )  # Select action based on returned action probabilities
        next_obs, reward, terminated, truncated, _ = env.step(action)

        episode_reward += reward
        step = EpisodeStep(obs, action)
        episode_steps.append(step)

        if terminated or truncated:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)

            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = next_obs


def filter_batches(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = np.mean(rewards)

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda s: s.observation, steps))
        train_act.extend(map(lambda s: s.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    # Add Episode Statistics to info variable
    env = wrappers.RecordEpisodeStatistics(env)

    # Run the agent one more time to record video
    env = gym.wrappers.RecordVideo(env, "recording")

    obs, _ = env.reset()

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(
        obs_size=obs_size,
        hidden_size=HIDDEN_SIZE,
        action_size=action_size,
    )

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(agent.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(
        iterate_batches(env=env, agent=agent, batch_size=BATCH_SIZE)
    ):
        obs_v, acts_v, reward_b, reward_m = filter_batches(batch, percentile=PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = agent(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        print(
            f"{iter_no} loss={loss_v.item():.6f} \
            reward_b={reward_b} \
            reward_mean={reward_m}"
        )
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        if reward_m > 499:
            print("Solved!")
            break

    writer.close()

    # Force Video Recording to finish
    env.close()

    # Run the agent one more time to record video
    env = gym.wrappers.RecordVideo(env, "recording")
    obs, _ = env.reset()

    while True:
        obs_v = torch.tensor([obs])
        action_probs_v = agent(obs_v)
        action_probs = action_probs_v.data.numpy()[0]
        action = np.argmax(action_probs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
