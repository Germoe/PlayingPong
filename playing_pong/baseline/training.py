import argparse
import sys
import time
from collections import deque, namedtuple

import numpy as np
import pong_env
import torch
import torch.nn as nn
import torch.optim as optim
from playing_pon.dqn_model import DQN
from torch.utils.tensorboard import SummaryWriter

ENV_NAME = "ALE/Pong-v5"
MEAN_REWARD_BOUND = 19.0

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

Experience = namedtuple(
    "Experience",
    ["state", "action", "reward", "terminated", "truncated", "new_state"],
)


class ExperienceBuffer:
    """
    Reference:
        - Code adjusted from
            `Lapan, Maxim. 2020. Deep Reinforcement Learning Hands-On:
            Apply Modern RL Methods to Practical Problems of Chatbots,
            Robotics, Discrete Optimization, Web Automation, and More.
            Packt Publishing Ltd. p.149`
    """

    def __init__(self, capacity, observation_shape, device="cpu"):
        if device == "cpu":
            self.buffer = deque(maxlen=capacity)
        else:
            # Use tensor for GPU computation
            self.capacity = capacity
            self.buffer = {
                "states": torch.empty(
                    (capacity, *observation_shape), dtype=torch.float32, device=device
                ),
                "actions": torch.empty((capacity, 1), dtype=torch.int64, device=device),
                "rewards": torch.empty(
                    (capacity, 1), dtype=torch.float32, device=device
                ),
                "terminated": torch.empty(
                    (capacity, 1), dtype=torch.bool, device=device
                ),
                "truncated": torch.empty(
                    (capacity, 1), dtype=torch.bool, device=device
                ),
                "new_states": torch.empty(
                    (capacity, *observation_shape), dtype=torch.float32, device=device
                ),
            }
        self.position = 0
        self.full = False
        self.device = device

    def __len__(self):
        if self.device == "cpu":
            return len(self.buffer)
        else:
            return self.capacity if self.full else self.position

    def append(self, experience):
        if self.device == "cpu":
            self.buffer.append(experience)
        else:
            self.buffer["states"][self.position] = torch.as_tensor(
                experience.state, dtype=torch.float32, device=self.device
            )
            self.buffer["actions"][self.position] = torch.as_tensor(
                [experience.action], dtype=torch.int64, device=self.device
            )
            self.buffer["rewards"][self.position] = torch.as_tensor(
                [experience.reward], dtype=torch.float32, device=self.device
            )
            self.buffer["terminated"][self.position] = torch.as_tensor(
                [experience.terminated], dtype=torch.bool, device=self.device
            )
            self.buffer["truncated"][self.position] = torch.as_tensor(
                [experience.truncated], dtype=torch.bool, device=self.device
            )
            self.buffer["new_states"][self.position] = torch.as_tensor(
                experience.new_state, dtype=torch.float32, device=self.device
            )

            self.position = (self.position + 1) % self.capacity
            self.full = self.full or self.position == 0

    def sample(self, batch_size):
        if self.device == "cpu":
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            states, actions, rewards, terminated, truncated, new_states = zip(
                *[self.buffer[idx] for idx in indices]
            )
            return (
                np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(terminated, dtype=np.uint8),
                np.array(truncated, dtype=np.uint8),
                np.array(new_states),
            )
        else:
            indices = torch.randint(0, len(self), (batch_size,), device=self.device)
            return (
                self.buffer["states"][indices],
                self.buffer["actions"][indices].view(-1),
                self.buffer["rewards"][indices].view(-1),
                self.buffer["terminated"][indices].view(-1),
                self.buffer["truncated"][indices].view(-1),
                self.buffer["new_states"][indices],
            )


class Agent:
    """
    Reference:
        - Code adjusted from
            `Lapan, Maxim. 2020. Deep Reinforcement Learning Hands-On:
            Apply Modern RL Methods to Practical Problems of Chatbots,
            Robotics, Discrete Optimization, Web Automation, and More.
            Packt Publishing Ltd. p.149`
    """

    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0

    def select_action(self, net, epsilon):
        if np.random.random() < epsilon:
            # Determine action randomly
            action = self.env.action_space.sample()
        else:
            # Determine Action using NN
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        return action

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        action = self.select_action(net, epsilon)

        new_state, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        exp = Experience(
            state=self.state,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            new_state=new_state,
        )
        self.exp_buffer.append(exp)
        self.state = new_state

        if terminated or truncated:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu") -> torch.Tensor:
    """
    Calculate the loss for a batch of experiences. The loss is calculated
    using the Bellman equation.

    Q(s,a) = (1-α) * Q(s,a) + α * (r + γ * max_a' Q(s',a'))

    The loss is calculated as the mean squared error between the Q-values
    of the current state and the Q-values of the next state.

    Args:
    batch: A batch of experiences.
    net: The neural network used to calculate the Q-values.
    tgt_net: The target network used to calculate the target Q-values.
    device: The device used for the calculations.

    Returns:
    The loss for the batch of experiences.

    Reference:
        - Code adjusted from
            `Lapan, Maxim. 2020. Deep Reinforcement Learning Hands-On:
            Apply Modern RL Methods to Practical Problems of Chatbots,
            Robotics, Discrete Optimization, Web Automation, and More.
            Packt Publishing Ltd. p.149`
    """
    states, actions, rewards, terminated, truncated, new_states = batch
    states_v = states  # copy=False to avoid unnecessary copy operation
    next_states_v = new_states
    actions_v = actions
    rewards_v = rewards
    terminated_mask = terminated
    truncated_mask = truncated

    """
    Select the Q-values of the action taken for each state.
    The gather-method extracts the column (i.e. dimension 1)
    from the net output values and actions_v determines the
    column index for each row (i.e. state) to be selected
    """
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    # Calculate the max_a Q(s',a')-values. max(1) returns (idx, val)
    # We are only interested in the value for now, thus `[0]`
    new_state_action_values = tgt_net(next_states_v).max(1)[0]

    # Terminated or truncated states do not have an associated reward value
    new_state_action_values[terminated_mask] = 0.0
    new_state_action_values[truncated_mask] = 0.0

    # Detach values from tgt_net computational graph (avoid gradient computation)
    new_state_action_values = new_state_action_values.detach()

    expected_state_action_value = new_state_action_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_value)


if __name__ == "__main__":
    """
    Reference:
        - Code adjusted from
            `Lapan, Maxim. 2020. Deep Reinforcement Learning Hands-On:
            Apply Modern RL Methods to Practical Problems of Chatbots,
            Robotics, Discrete Optimization, Web Automation, and More.
            Packt Publishing Ltd. p.153-155`
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        default=False,
        action="store_true",
        help="Enable GPU Computation via Cuda",
    )
    parser.add_argument(
        "--env",
        default=ENV_NAME,
        help="Name of the Pong Environment Default: " + ENV_NAME,
    )
    parser.add_argument(
        "--record",
        default=False,
        action="store_true",
        help="Record a video of the agent playing",
    )
    parser.add_argument(
        "--record_at_iter",
        default=100,
        type=int,
        help="Record a video of the agent playing every n iterations",
    )
    parser.add_argument(
        "--load",
        default=None,
        type=str,
        help="Load a pretrained model from the specified path",
    )
    parser.add_argument(
        "--epsilon_start",
        default=EPSILON_START,
        type=float,
        help="Set the starting epsilon value",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Using device: ", device)

    env = pong_env.make_env(args.env, args.record, args.record_at_iter)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)

    if args.load is not None:
        net.load_state_dict(torch.load(args.load))
        tgt_net.load_state_dict(torch.load(args.load))
        print(f"Loaded model from {args.load}")

    writer = SummaryWriter()

    buffer = ExperienceBuffer(REPLAY_SIZE, env.observation_space.shape, device=device)
    agent = Agent(env, buffer)
    epsilon = args.epsilon_start
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    # last_recorded_iter = -1
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None
    last_m_reward = None
    update_frame = -1

    while True:
        frame_idx += 1
        epsilon = max(
            EPSILON_FINAL, args.epsilon_start - frame_idx / EPSILON_DECAY_LAST_FRAME
        )
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])

            if best_m_reward is None or m_reward > best_m_reward:
                torch.save(
                    net.state_dict(),
                    args.env.replace("/", "_") + f"-best_{m_reward:.0f}.dat",
                )
                last_m_reward = best_m_reward if best_m_reward is not None else m_reward
                best_m_reward = m_reward
                update_frame = frame_idx

            sys.stdout.write(
                f"\r{frame_idx}: done {len(total_rewards)} games, "
                f"reward {m_reward:.3f}, "
                f"eps {epsilon:.2f}, speed {speed:.2f} f/s, "
                f"Best reward updated at {update_frame} {last_m_reward:.2f} "
                f"-> {best_m_reward:.2f}"
            )
            sys.stdout.flush()
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_m_reward > MEAN_REWARD_BOUND:
                print(f"Solved in {frame_idx} frames!")
                break

        if len(buffer) < REPLAY_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()

        batch_x = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch_x, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
