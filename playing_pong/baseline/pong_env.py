import datetime
import os

import cv2
import gymnasium as gym
import gymnasium.wrappers as wrappers
import numpy as np

RECORD = True
RECORD_AT_ITER = 10


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
        - Code adjusted from
            `Lapan, Maxim. 2020. Deep Reinforcement Learning Hands-On:
            Apply Modern RL Methods to Practical Problems of Chatbots,
            Robotics, Discrete Optimization, Web Automation, and More.
            Packt Publishing Ltd. p.143-144`

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

    Reference:
        `Lapan, Maxim. 2020. Deep Reinforcement Learning Hands-On:
        Apply Modern RL Methods to Practical Problems of Chatbots,
        Robotics, Discrete Optimization, Web Automation, and More.
        Packt Publishing Ltd. p.145-146`

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

    Reference:
        - Code adjusted from
            `Lapan, Maxim. 2020. Deep Reinforcement Learning Hands-On:
            Apply Modern RL Methods to Practical Problems of Chatbots,
            Robotics, Discrete Optimization, Web Automation, and More.
            Packt Publishing Ltd. p.144`
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

    Reference:
        - Code adjusted from
            `Lapan, Maxim. 2020. Deep Reinforcement Learning Hands-On:
            Apply Modern RL Methods to Practical Problems of Chatbots,
            Robotics, Discrete Optimization, Web Automation, and More.
            Packt Publishing Ltd. p.143-144`
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

    Reference:
        - Code adjusted from
            `Lapan, Maxim. 2020. Deep Reinforcement Learning Hands-On:
            Apply Modern RL Methods to Practical Problems of Chatbots,
            Robotics, Discrete Optimization, Web Automation, and More.
            Packt Publishing Ltd. p.144`
    """

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name, record=False, record_at_iter=10):
    env = gym.make(env_name, render_mode="rgb_array")
    env = wrappers.RecordEpisodeStatistics(
        env
    )  # Standard Wrapper to record episode statistics in info
    if record is True:
        print(f"Record Videos: {record}, at every {record_at_iter}th episode")
        # Create a folder to store the videos
        path = "./videos/{}".format(
            env_name.replace("/", "_")
            + "_"
            + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
        )
        if not os.path.exists(path):
            os.makedirs(path)
        env = wrappers.RecordVideo(
            env,
            path,
            episode_trigger=lambda x: x % record_at_iter == 0,
            disable_logger=True,
        )  # Standard Wrapper to record videos of episodes
    env = PreProcessingImage84(env)
    env = ActionRepeated(env, repeat=3)
    env = ImageToCWH(env)
    env = Buffer(env, size=3)
    env = NormImage(env)
    return env
