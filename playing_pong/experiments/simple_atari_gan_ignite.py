import argparse
from datetime import datetime

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from torch.utils.tensorboard import SummaryWriter

log = gym.logger
log.set_level(gym.logger.INFO)

LATENT_VECTOR_SIZE = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

REPORT_EVERY_ITER = 10
SAVE_IMAGE_EVERY_ITER = 100

# Number of filters
DISCR_FILTERS = 64
GENER_FILTERS = 64

# Dimensions of the image
IMAGE_SIZE = 64


class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input numpy array:
    1. resize image into predefined size
    2. move color channel axis to a first place
    """

    def __init__(self, *args):
        super().__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            self.observation(
                old_space.low
            ),  # Apply new observation to provide correct shape
            self.observation(
                old_space.high
            ),  # Apply new observation to provide correct shape
            dtype=np.float32,
        )

    def observation(self, observation):
        # resize observation to IMAGE_SIZE square
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))

        # shift axis from (210, 160, 3) to (3, 210, 160)
        # convention for convolutional layers
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=DISCR_FILTERS,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=DISCR_FILTERS,
                out_channels=DISCR_FILTERS * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(DISCR_FILTERS * 2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=DISCR_FILTERS * 2,
                out_channels=DISCR_FILTERS * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=DISCR_FILTERS * 4,
                out_channels=DISCR_FILTERS * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=DISCR_FILTERS * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=GENER_FILTERS * 8,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=GENER_FILTERS * 8,
                out_channels=GENER_FILTERS * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=GENER_FILTERS * 4,
                out_channels=GENER_FILTERS * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=GENER_FILTERS * 2,
                out_channels=GENER_FILTERS,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=GENER_FILTERS,
                out_channels=output_shape[0],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.pipe(x)


def iterate_batches(env, batch_size=BATCH_SIZE):
    obs, _ = env.reset()
    batch = []

    while True:
        obs, _, terminated, _, _ = env.step(env.action_space.sample())
        batch.append(obs)
        if len(batch) == batch_size:
            # Normalize Input between -1 and 1
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if terminated:
            env.reset()


def process_batch(engine, batch) -> tuple[float, float]:
    batch_v = batch.to(device)

    # Generate a random vector and pass it to the generator network
    gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
    gen_input_v.normal_(0, 1)
    gen_input_v = gen_input_v.to(device)
    gen_output_v = gener_net(gen_input_v)

    # First pass batch and generated images through discriminator for training
    dis_optimizer.zero_grad()
    dis_output_true_v = discr_net(batch_v)
    dis_output_fake_v = discr_net(
        gen_output_v.detach()
    )  # .detach() makes a copy without connections to the parent
    dis_loss = loss(dis_output_true_v, true_labels_v) + loss(
        dis_output_fake_v, fake_labels_v
    )
    dis_loss.backward()
    dis_optimizer.step()

    # Second pass batch through generator for training
    gen_optimizer.zero_grad()
    dis_output_v = discr_net(gen_output_v)
    gen_loss = loss(dis_output_v, true_labels_v)
    gen_loss.backward()
    gen_optimizer.step()

    if engine.state.iteration % SAVE_IMAGE_EVERY_ITER == 0:
        fake_img = vutils.make_grid(gen_output_v.data[:64], normalize=True)
        engine.tb.writer.add_image("fake", fake_img, engine.state.iteration)

        real_img = vutils.make_grid(batch_v.data[:64], normalize=True)
        engine.tb.writer.add_image("real", real_img, engine.state.iteration)
        engine.tb.writer.flush()

    return dis_loss.item(), gen_loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable cuda computation"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    envs = [
        InputWrapper(gym.make(name, render_mode="rgb_array"))
        for name in ["ALE/Pong-v5"]
    ]

    input_shape = envs[0].observation_space.shape

    discr_net = Discriminator(input_shape=input_shape).to(device)
    gener_net = Generator(output_shape=input_shape).to(device)

    loss = nn.BCELoss()

    gen_optimizer = optim.Adam(
        gener_net.parameters(), lr=LEARNING_RATE * 10, betas=(0.5, 0.999)
    )
    dis_optimizer = optim.Adam(
        discr_net.parameters(), lr=LEARNING_RATE / 10, betas=(0.5, 0.999)
    )
    writer = SummaryWriter()

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    engine = Engine(process_batch)
    # Initialize and Register TB Logger
    tb = tb_logger.TensorboardLogger(log_dir=f"runs/{datetime.now()}")
    engine.tb = tb

    # Write running averages of training process
    RunningAverage(output_transform=lambda out: out[0]).attach(engine, "avg_loss_dis")
    RunningAverage(output_transform=lambda out: out[1]).attach(engine, "avg_loss_gen")

    handler = tb_logger.OutputHandler(
        tag="train", metric_names=["avg_loss_dis", "avg_loss_gen"]
    )
    tb.attach(engine, log_handler=handler, event_name=Events.ITERATION_COMPLETED)

    @engine.on(Events.ITERATION_COMPLETED)
    def log_losses(engine):
        if engine.state.iteration % REPORT_EVERY_ITER == 0:
            log.info(
                f"""{engine.state.iteration} \
                    - Gen Loss: {engine.state.metrics['avg_loss_gen']:.8f} \
                    Dis Loss: {engine.state.metrics['avg_loss_dis']:.8f}"""
            )

    engine.run(iterate_batches(envs[0]))