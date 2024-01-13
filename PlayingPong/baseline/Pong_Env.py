import argparse

import cv2
import gymnasium as gym
import gymnasium.wrappers as wrappers
import numpy as np

RECORD = False
RECORD_AT_ITER = 1000


class PreProcessingImage84(gym.ObservationWrapper):
    """
    This class is a wrapper for a gym environment that preprocesses the
    observation images. It inherits from gym.ObservationWrapper, which is
    a base class for wrapping gym environments to modify their observations.

    The class rescales, crops, and converts the observation images to grayscale.

    Attributes:
    env: The gym environment to be wrapped.
    observation_space: The observation space of the gym environment.

    Reference:
    - TODO: Add Paper and Hands-On Book

    Note:
    - Manual Implementation some methods in this Wrapper Class `AtariPreprocessing`:
    https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.AtariPreprocessing
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(84, 84, 1), dtype=np.uint8
        )

    @staticmethod
    def convert_to_grayscale(obs):
        return (
            obs[:, :, 0] * 0.299 + obs[:, :, 1] * 0.587 + obs[:, :, 2] * 0.114
        ).astype(obs.dtype)

    @staticmethod
    def rescale(obs, dim):
        return cv2.resize(obs, dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def crop(obs, y1, y2, x1, x2):
        return obs[y1:y2, x1:x2]

    def observation(self, obs):
        obs = self.rescale(obs, dim=(84, 110))
        obs = self.crop(obs, y1=18, y2=102, x1=0, x2=84)
        return self.convert_to_grayscale(obs).astype(np.uint8).reshape(84, 84, 1)


class ActionRepeated(gym.Wrapper):
    """
    This class is a wrapper for a gym environment, allowing an action to
    be repeated a specified number of times. It inherits from gym.Wrapper,
    which is a base class for wrapping gym environments to modify their behavior.

    The class overrides the step() method of the gym environment to repeat
    the action for a given number of times, accumulating the reward for each
    repetition. If the environment is terminated or truncated during the repetitions,
    the loop breaks and the method returns the observation, total reward,
    termination status, truncation status, and info.

    Attributes:
    env: The gym environment to be wrapped.
    repeat: The number of times an action is repeated.
    """

    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class ImageToCWH(gym.ObservationWrapper):
    """
    This class is a wrapper for a gym environment, that converts the
    observation images from HWC to CWH format.

    Attributes:
    env: The gym environment to be wrapped.
    observation_space: The observation space of the gym environment.
    """

    def __init__(self, env):
        super().__init__(env)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_space.shape[-1], old_space.shape[0], old_space.shape[1]),
            dtype=np.float32,
        )

    def observation(self, obs):
        return np.moveaxis(obs, 2, 0)


class Buffer(gym.ObservationWrapper):
    """
    This class is a wrapper for a gym environment, that stacks the
    observations in a buffer.

    Attributes:
    env: The gym environment to be wrapped.
    size: The size of the buffer.
    dtype: The data type of the buffer.
    """

    def __init__(self, env, size, dtype=np.float32):
        super().__init__(env)
        self.size, self.dtype = size, dtype
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            low=old_space.low.repeat(size, axis=0),
            high=old_space.high.repeat(size, axis=0),
            dtype=dtype,
        )

    def reset(self, **kwargs):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        self.buffer[:-1] = self.buffer[
            1:
        ]  # Buffer stacked observations (first in, first out)
        self.buffer[-1] = obs
        return self.buffer


class NormImage(gym.ObservationWrapper):
    """
    This class is a wrapper for a gym environment, that normalizes
    the observation images.

    Attributes:
    env: The gym environment to be wrapped.
    """

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    env = wrappers.RecordEpisodeStatistics(
        env
    )  # Standard Wrapper to record episode statistics in info
    if RECORD is True:
        env = wrappers.RecordVideo(
            env, directory="videos", episode_trigger=lambda x: x % RECORD_AT_ITER == 0
        )  # Standard Wrapper to record videos of episodes
    env = PreProcessingImage84(env)
    env = ActionRepeated(env)
    env = ImageToCWH(env)
    env = Buffer(env, size=4)
    env = NormImage(env)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ALE/Pong-v5")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    env_name = args.env
    steps = args.steps

    env = make_env(env_name)
    obs, _ = env.reset()

    # Play n steps
    for i in range(steps):
        action = env.action_space.sample()
        next_obs, _, _, _, _ = env.step(action)

        # Grayscale Observation Human Eye
        cv2.imwrite(
            "screenshots/pong_{}.png".format(i),
            np.reshape(next_obs * 255.0, (-1, 84)),
        )
