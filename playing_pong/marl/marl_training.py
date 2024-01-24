import argparse
import datetime
import os
import sys
import time

import marl_pong_env
import numpy as np

# Import ReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb
from marl_dqn_model import DQN
from torchrl.data import ListStorage, ReplayBuffer

REPLAY_SIZE = 10000
BATCH_SIZE = 32

EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_LAST_FRAME = 100000

LEARNING_RATE = 1e-4
TARGET_NET_SYNC_RATE = 1000

GAMMA = 0.99


class Agent:
    def __init__(self, agent_id, epsilon_start, epsilon_end, epsilon_decay_last_frame):
        self.agent_id = agent_id
        self.exp_buffer = ReplayBuffer(
            storage=ListStorage(max_size=REPLAY_SIZE), batch_size=BATCH_SIZE
        )
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_last_frame = epsilon_decay_last_frame

        self._action = None
        self._obs = None
        self._iter = 0
        self._loss = 0.0
        self.rewards = []
        self.curr_reward = 0.0

    def epsilon(self):
        """
        Method to calculate the epsilon value for the current iteration.
        It uses a linear decay function to calculate the epsilon value.

        Args:
            None

        Returns:
            The epsilon value for the current iteration.
        """
        return self.epsilon_start * (
            1 - self._iter / self.epsilon_decay_last_frame
        ) + self.epsilon_end * (self._iter / self.epsilon_decay_last_frame)

    def reset_reward(self):
        self.curr_reward = 0.0

    def update_reward(self, reward):
        self.curr_reward += reward

    def append_reward(self, reward):
        self.rewards.append(reward)

    def select_action(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class DQNAgent(Agent):
    def __init__(
        self,
        env,
        agent_id: str,
        epsilon_start,
        epsilon_end,
        epsilon_decay_last_frame,
        device,
        load=False,
    ):
        super().__init__(agent_id, epsilon_start, epsilon_end, epsilon_decay_last_frame)
        self.env = env
        self.device = device
        self.net = DQN(
            env.observation_space(agent_id).shape, env.action_space(agent_id).n
        ).to(device)
        if load is not False:
            self.net.load_state_dict(torch.load(load, map_location=device))
            print(f"Loaded model from {load}")

        self.tgt_net = DQN(
            env.observation_space(agent_id).shape, env.action_space(agent_id).n
        ).to(device)
        self.tgt_net.load_state_dict(self.net.state_dict())
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self._tgt_net_sync_rate = TARGET_NET_SYNC_RATE

    @torch.no_grad()
    def select_action(self, obs):
        """
        Reference:
            `Lapan, Maxim. 2020. Deep Reinforcement Learning Hands-On:
            Apply Modern RL Methods to Practical Problems of Chatbots,
            Robotics, Discrete Optimization, Web Automation, and More.
            Packt Publishing Ltd. p.143`
        """
        if np.random.random() < self.epsilon():
            # Determine action randomly
            action = self.env.action_space(self.agent_id).sample()
        else:
            # Determine Action using NN
            state_a = np.array([obs], copy=False)
            state_v = torch.tensor(state_a).to(self.device)
            q_vals_v = self.net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # Save action for later retrieval
        self._action = action
        self._obs = obs
        return action

    def add_exp(self, obs, action, reward, terminated, truncated, next_obs):
        self.exp_buffer.add((obs, action, reward, terminated, truncated, next_obs))

    def sample(self):
        (
            obs,
            actions,
            rewards,
            terminations,
            truncations,
            next_obs,
        ) = self.exp_buffer.sample()

        obs_v = convert_to_tensor(obs, dtype=torch.float32)
        actions_v = convert_to_tensor(actions, dtype=torch.int64)
        rewards_v = convert_to_tensor(rewards, dtype=torch.float32)
        terminations_v = convert_to_tensor(terminations, dtype=torch.bool)
        truncations_v = convert_to_tensor(truncations, dtype=torch.bool)
        next_obs_v = convert_to_tensor(next_obs, dtype=torch.float32)

        return (
            obs_v,
            actions_v.view(-1),
            rewards_v.view(-1),
            terminations_v.view(-1),
            truncations_v.view(-1),
            next_obs_v,
        )

    def sync_tgt_net(self):
        self.tgt_net.load_state_dict(self.net.state_dict())

    def update(self):
        """
        Reference:
            - Adapted from code in
            `Lapan, Maxim. 2020. Deep Reinforcement Learning Hands-On:
            Apply Modern RL Methods to Practical Problems of Chatbots,
            Robotics, Discrete Optimization, Web Automation, and More.
            Packt Publishing Ltd. p.143`
        """
        if len(self.exp_buffer) < REPLAY_SIZE:
            return

        # Keeping the Target Network in sync with the Main Network
        self._iter += 1
        if self._iter % self._tgt_net_sync_rate == 0:
            self.sync_tgt_net()

        # Update the network parameters
        self.optimizer.zero_grad()
        loss_t = self.calc_loss()
        self._loss = loss_t.item()
        loss_t.backward()
        self.optimizer.step()

    def calc_loss(self) -> torch.Tensor:
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
        (
            states_v,
            actions_v,
            rewards_v,
            terminated_mask,
            truncated_mask,
            next_states_v,
        ) = self.sample()

        """
        Select the Q-values of the action taken for each state.
        The gather-method extracts the column (i.e. dimension 1)
        from the net output values and actions_v determines the
        column index for each row (i.e. state) to be selected
        """

        state_action_values = (
            self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        )

        # Calculate the max_a Q(s',a')-values. max(1) returns (idx, val)
        # We are only interested in the value for now, thus `[0]`
        new_state_action_values = self.tgt_net(next_states_v).max(1)[0]

        # Terminated or truncated states do not have an associated reward value
        new_state_action_values[terminated_mask] = 0.0
        new_state_action_values[truncated_mask] = 0.0

        # Detach values from tgt_net computational graph (avoid gradient computation)
        new_state_action_values = new_state_action_values.detach()

        expected_state_action_value = new_state_action_values * GAMMA + rewards_v
        return nn.MSELoss()(state_action_values, expected_state_action_value)


def convert_to_tensor(input, dtype=torch.float32):
    return torch.as_tensor(input, dtype=dtype, device=device)


class MARL_DQN_Algorithm:
    def __init__(self, env, agents):
        self.env = env
        self.agents = agents
        self.train_agents = {a_id: True for a_id in self.agents}
        self._reset()

    def _reset(self):
        self.env.reset()
        self.agent_iter = iter(self.env.agent_iter())

    def step(self):
        """
        The step function is the main env sampling function for the
        algorithm.

        If no agent is active, the environment is reset and the function
        returns. Otherwise, the active agent selects an action and the
        environment is stepped.
        """

        try:
            a_id = next(self.agent_iter)
        except StopIteration:
            self._reset()
            return None

        next_obs, reward, terminated, truncated, _ = self.env.last()

        active_a = self.agents[a_id]
        active_a.update_reward(reward)

        if a_id is not None:
            if self.train_agents[a_id]:
                self.update_agent(a_id, reward, next_obs, terminated, truncated)

        if terminated or truncated:
            active_a.append_reward(active_a.curr_reward)
            active_a.reset_reward()
            action = None
        else:
            action = active_a.select_action(next_obs)

        self.env.step(action)

        return a_id

    def set_train_agents(self, train_agents):
        self.train_agents = train_agents

    def update_agent(self, agent_id, reward, next_obs, terminated, truncated):
        active_a = self.agents[agent_id]
        if active_a._action is not None:
            # Add experience and take a training step
            active_a.add_exp(
                active_a._obs,
                active_a._action,
                reward,
                terminated,
                truncated,
                next_obs,
            )
            active_a.update()


if __name__ == "__main__":
    """
    Sample run command:

    python marl_training.py --cuda \
        --agent0 path/to/agent0/model \
        --agent1 path/to/agent1/model \
        --epsilon 1.0 0.1 100000
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable CUDA"
    )
    parser.add_argument(
        "--agent0",
        default=False,
        type=str,
        help="Load model weights from model_weights folder",
    )
    parser.add_argument(
        "--agent1",
        default=False,
        type=str,
        help="Load model weights from model_weights folder",
    )
    parser.add_argument(
        "--epsilon",
        default=(EPSILON_START, EPSILON_END, EPSILON_DECAY_LAST_FRAME),
        nargs=3,
        type=float,
        help="Epsilon start, end and decay last frame as a tuple",
    )

    args = parser.parse_args()
    epsilon_start, epsilon_end, epsilon_decay_last_frame = args.epsilon
    print("Epsilon: ", epsilon_start, epsilon_end, epsilon_decay_last_frame)
    agents_to_load = {0: args.agent0, 1: args.agent1}
    env = marl_pong_env.make_env(record=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    run_name = f"pong-marl-{datetime.datetime.now().isoformat()}".replace(":", "_")
    print("Using device: ", device)

    agents = {}
    for i, agent_id in enumerate(env.possible_agents):
        agents[agent_id] = DQNAgent(
            env,
            agent_id,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_last_frame=epsilon_decay_last_frame,
            device=device,
            load=agents_to_load[i],
        )

    n_agents = len(agents)

    marl = MARL_DQN_Algorithm(env, agents)
    iteration = 0
    episodes = 0
    episode_last_iter_n = 0
    last_iteration = 0
    last_save_iteration = 0
    ts_time = time.time()
    mean_reward_10 = {a_id: -100 for a_id in agents}
    mean_reward_100 = {a_id: -100 for a_id in agents}

    writer = tb.SummaryWriter(comment="-pong-marl")

    while True:
        iteration += 1
        speed = (iteration - last_iteration) / (time.time() - ts_time)
        last_iteration = iteration
        ts_time = time.time()

        a_id = marl.step()
        if a_id is None:
            episodes += 1
            # Marks the end of an episode

            a_ids = list(marl.env.possible_agents)
            for a_id in a_ids:
                mean_reward_100[a_id] = np.mean(marl.agents[a_id].rewards[-100:])
                mean_reward_10[a_id] = np.mean(marl.agents[a_id].rewards[-10:])

                writer.add_scalar(
                    f"mean-reward-100/{a_id}",
                    mean_reward_100[a_id],
                    iteration / n_agents,
                )

                writer.add_scalar(
                    f"mean-reward-10/{a_id}",
                    mean_reward_10[a_id],
                    iteration / n_agents,
                )

            # Occasionally save the agent
            if episodes % 50 == 0 and episodes > 0:
                # Check if directory exists
                if not os.path.exists(f"model_weights/{run_name}"):
                    os.makedirs(f"model_weights/{run_name}")

                for a_id in marl.env.possible_agents:
                    torch.save(
                        marl.agents[a_id].net.state_dict(),
                        f"./model_weights/{run_name}/pong-marl-{a_id}-{episodes}-{mean_reward_10[a_id]:.0f}.dat",
                    )

            mean_rewards = [f"{mean_reward_100[a_id]:.2f}" for a_id in a_ids]
            mean_rewards_10 = [f"{mean_reward_10[a_id]:.2f}" for a_id in a_ids]

            # If the mean reward is overpowering, stop training for that agent
            for a_id in a_ids:
                if marl.agents[a_id].epsilon() == EPSILON_END:
                    if mean_reward_100[a_id] is not None:
                        if (
                            mean_reward_100[a_id] > 10
                            and marl.train_agents[a_id] is True
                        ):
                            print(
                                f"Stopping training for agent {a_id}, "
                                "others continue/restart"
                            )
                            # Update marl.train_agents dict (set other agents to True)
                            new_train_agents = {a_id: True for a_id in a_ids}
                            new_train_agents[a_id] = False
                            marl.set_train_agents(new_train_agents)
                        if (
                            mean_reward_100[a_id] < -5
                            and marl.train_agents[a_id] is False
                        ):
                            print(
                                f"Restarting training for agent {a_id}, others continue"
                            )
                            new_train_agents = marl.train_agents
                            new_train_agents[a_id] = True
                            marl.set_train_agents(new_train_agents)

            epsilons = [f"{marl.agents[a_id].epsilon():.3f}" for a_id in a_ids]
            sys.stdout.write(
                f"\rEpisodes: {episodes} | "
                f"Iterations (last): {iteration - episode_last_iter_n} | "
                f"Rw. 100: {mean_rewards} | "
                f"Rw. 10: {mean_rewards_10} | "
                f" {speed:.2f} iter/s | "
                f"Eps: {epsilons}   "
            )

            writer.add_scalar(
                "episode_iterations", iteration - episode_last_iter_n, episodes
            )

            episode_last_iter_n = iteration

        if iteration % 100 == 0:
            obs, _, _, _, _ = marl.env.last()

            writer.add_image(
                f"screen/agent-{a_id}",
                obs.astype(np.float32),
                iteration / n_agents,
                dataformats="CHW",
            )

        writer.add_scalar(
            f"epsilon/{a_id}", agents[a_id].epsilon(), iteration / n_agents
        )
        writer.add_scalar(f"speed/{a_id}", speed, iteration / n_agents)

        writer.add_scalar(f"loss/{a_id}", agents[a_id]._loss, iteration / n_agents)
