from collections import namedtuple

import gymnasium as gym
import gymnasium.wrappers as wrappers
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

LOG_AT_ITER = 100
ENV_NAME = "FrozenLake-v1"
MAP_NAME = "4x4"

GAMMA = 0.95
HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 20
LEARNING_RATE = 0.001

Episode = namedtuple("Episode", ["reward", "steps"])
EpisodeStep = namedtuple("EpisodeStep", ["observation", "action"])


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        shape = (env.observation_space.n,)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape, dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


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
        action_probs = action_probs_v.data.numpy()[0]
        action = np.random.choice(len(action_probs), p=action_probs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        episode_reward += reward
        episode_step = EpisodeStep(obs, action)
        episode_steps.append(episode_step)

        if terminated or truncated:
            e = Episode(episode_reward, episode_steps)
            batch.append(e)

            # Reset environment to initial state
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = next_obs


def filter_batches(
    batch, percentile
) -> tuple[list, torch.Tensor, torch.Tensor, float, float]:
    disc_rewards = list(map(lambda b: b.reward * (GAMMA ** len(b.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)
    reward_mean = np.mean(disc_rewards)

    train_obs: list[np.array] = []
    train_acts: list[np.array] = []
    elite_batch = []

    for episode, disc_reward in zip(batch, disc_rewards):
        if disc_reward <= reward_bound:
            continue
        train_obs.extend(map(lambda s: s.observation, episode.steps))
        train_acts.extend(map(lambda s: s.action, episode.steps))
        elite_batch.append(episode)

    train_obs_v = torch.FloatTensor(train_obs)
    train_acts_v = torch.LongTensor(train_acts)

    return elite_batch, train_obs_v, train_acts_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make(
        ENV_NAME,
        desc=None,
        map_name=MAP_NAME,
        is_slippery=True,
        render_mode="rgb_array",
    )
    env = ObservationWrapper(env)

    # Log Episode Statistics
    env = wrappers.RecordEpisodeStatistics(env)

    writer = SummaryWriter(comment="-frozenlake")

    obs, _ = env.reset()
    obs_size = env.observation_space.shape[0]
    hidden_size = HIDDEN_SIZE
    action_size = env.action_space.n

    agent = Agent(obs_size=obs_size, hidden_size=hidden_size, action_size=action_size)

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    elite_batch: list[Episode] = []  # List of elite episodes
    for iter_no, batch in enumerate(iterate_batches(env, agent, batch_size=BATCH_SIZE)):
        elite_batch, train_obs_v, train_acts_v, rw_bound, reward_m = filter_batches(
            elite_batch + batch, percentile=PERCENTILE
        )

        if len(elite_batch) == 0:
            continue

        elite_batch = elite_batch[-500:]

        optimizer.zero_grad()
        action_scores_v = agent(train_obs_v)
        loss_v = objective(action_scores_v, train_acts_v)
        loss_v.backward()
        optimizer.step()

        if iter_no % LOG_AT_ITER == 0:
            print(
                f"{iter_no} loss={loss_v} \
                rw_bound={rw_bound} \
                reward_mean={reward_m} \
                elite_batch={len(elite_batch)}"
            )

        writer.add_scalar("loss", loss_v, iter_no)
        writer.add_scalar("reward_bound", rw_bound, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        if reward_m > 0.8:
            print("Solved!")
            break

        # # Save video of the agent playing the game
        # if iter_no % (LOG_AT_ITER) == 0:
        #     rec_env = gym.wrappers.RecordVideo(env, "recording")
        #     sm = nn.Softmax(dim=1)
        #     obs, _ = rec_env.reset()

        #     while True:
        #         obs_v = torch.FloatTensor([obs])
        #         action_probs_v = sm(agent(obs_v))
        #         action_probs = action_probs_v.data.numpy()[0]
        #         action = np.random.choice(len(action_probs), p=action_probs)
        #         obs, _, terminated, _, _ = rec_env.step(action)

        #         if terminated:
        #             break

        #     rec_env.close()
