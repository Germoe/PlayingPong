import os
from datetime import datetime

import cv2
import gymnasium as gym
import numpy as np
from pettingzoo.atari import pong_v3
from pettingzoo.utils.wrappers import BaseWrapper
from supersuit import (
    color_reduction_v0,
    dtype_v0,
    frame_skip_v0,
    frame_stack_v1,
    normalize_obs_v0,
    resize_v1,
)

MAX_CYCLES_PER_EPISODE = 100000
RECORD_EVERY = 10


class shift_channel(BaseWrapper):
    """
    Custom reshaping wrapper for Pong Environment in PettingZoo.

    This wrapper shifts the channel dimension to the first dimension. As
    PyTorch expects the channel dimension to be the first dimension.

    Attributes:
        env: The gym environment to be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)
        self.old_space = {
            agent: self.env.observation_space(agent) for agent in self.possible_agents
        }

    def observation_space(self, agent):
        """
        Adjust the observation space of the environment for each agent. Takes the
        dtype, high and low of the old observation space but updates the shape
        to the crop size.
        """
        # Minimum and maximum values of the observation space
        low = np.min(self.old_space[agent].low)
        high = np.max(self.old_space[agent].high)
        new_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(
                self.old_space[agent].shape[2],
                self.old_space[agent].shape[0],
                self.old_space[agent].shape[1],
            ),
            dtype=self.old_space[agent].dtype,
        )
        return new_space

    def observe(self, agent):
        obs = self.env.observe(agent)
        return np.moveaxis(obs, 2, 0)


class video_wrapper(BaseWrapper):
    def __init__(self, env, every=10, path="videos"):
        super().__init__(env)
        self.episode = -1
        self.every = every
        self.frames = []
        self.space = self.env.observation_space(self.env.possible_agents[0])
        self.width = self.space.shape[0]
        self.height = self.space.shape[1]
        self.path = path

    def reset(self, *args, **kwargs):
        if self.episode % self.every == 0:
            print(f"Saving video for episode {self.episode}")
            self.save_video()

        self.episode += 1
        return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        action = args[0]
        res = self.env.step(*args, **kwargs)
        if self.episode % self.every == 0 and action is not None:
            obs, _, _, _, _ = self.env.last()
            self.frames.append(obs)
        return res

    def save_video(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ts = datetime.now().isoformat()[:-7].replace(":", "_")
        out = cv2.VideoWriter(
            f"{self.path}/{self.episode}-{ts}.mp4",
            fourcc,
            30,
            (self.height, self.width),
        )

        for rgb_array in self.frames:
            # Check if rgb_array is valid type (np.uint8) and range (0,255)
            if not isinstance(rgb_array, np.ndarray):
                rgb_array = np.asarray(rgb_array)
            if not rgb_array.dtype == np.uint8:
                rgb_array = rgb_array.astype(np.uint8)
            if not np.max(rgb_array) <= 255:
                rgb_array = rgb_array / np.max(rgb_array) * 255

            # Convert RGB to BGR
            bgr_frame = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

            # Write the frame into the file 'output.mp4'
            out.write(bgr_frame)

        # Release everything when the job is finished
        out.release()
        self.frames = []


class crop_wrapper(BaseWrapper):
    """
    Custom Crop Wrapper for Pong Environment in PettingZoo.

    Attributes:
        env: The gym environment to be wrapped.
        y1: The top y coordinate of the crop.
        y2: The bottom y coordinate of the crop.
        x1: The left x coordinate of the crop.
        x2: The right x coordinate of the crop.
    """

    def __init__(self, env, y1, y2, x1, x2):
        super().__init__(env)
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2
        self.old_space = {
            agent: self.env.observation_space(agent) for agent in self.possible_agents
        }

    def observation_space(self, agent):
        """
        Adjust the observation space of the environment for each agent. Takes the
        dtype, high and low of the old observation space but updates the shape
        to the crop size.
        """
        # Minimum and maximum values of the observation space
        low = np.min(self.old_space[agent].low)
        high = np.max(self.old_space[agent].high)
        new_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(self.y2 - self.y1, self.x2 - self.x1, 1),
            dtype=self.old_space[agent].dtype,
        )
        return new_space

    def observe(self, agent):
        obs = self.env.observe(agent)
        return obs[self.y1 : self.y2, self.x1 : self.x2]


def make_env(
    record=False, record_every=RECORD_EVERY, max_cycles=MAX_CYCLES_PER_EPISODE
):
    env = pong_v3.env(render_mode="rgb_array", max_cycles=max_cycles)
    env = color_reduction_v0(env, mode="R")
    env = dtype_v0(env, dtype=np.float32)
    env = resize_v1(env, x_size=84, y_size=110)
    env = crop_wrapper(env, y1=18, y2=102, x1=0, x2=84)
    env = frame_skip_v0(env, num_frames=3)
    if record is True:
        env = video_wrapper(env, every=record_every, path="videos")
    env = frame_stack_v1(env, stack_size=3)
    env = shift_channel(env)
    env = normalize_obs_v0(env, env_min=0, env_max=1)

    return env
