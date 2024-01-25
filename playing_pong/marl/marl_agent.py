import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from marl_dqn_model import DQN
from torchrl.data import ListStorage, ReplayBuffer

REPLAY_SIZE = 50000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
TARGET_NET_SYNC_RATE = 1000

GAMMA = 0.99


class Agent:
    def __init__(self, agent_id, epsilon_start, epsilon_end, epsilon_decay_last_frame):
        self.agent_id = agent_id
        self.replay_size = REPLAY_SIZE
        self.exp_buffer = ReplayBuffer(
            storage=ListStorage(max_size=self.replay_size), batch_size=BATCH_SIZE
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
        alpha = min(1, self._iter / self.epsilon_decay_last_frame)
        return (
            self.epsilon_start * (1 - alpha) + self.epsilon_end * alpha
        )  # Linear decay

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

    def convert_to_tensor(self, input, dtype=torch.float32):
        return torch.as_tensor(input, dtype=dtype, device=self.device)

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
            _, action = self.q_max(obs)

        # Save action and observation for later retrieval
        self._action = action
        self._obs = obs
        return action

    def q(self, obs) -> torch.Tensor:
        """
        Predict the Q values for the given observations
        """
        state_a = np.array([obs], copy=False)
        state_v = torch.tensor(state_a).to(self.device)
        return self.net(state_v)

    def q_max(self, obs) -> tuple[float, float]:
        """
        Predict the Q values for the given observations
        """
        q_max_val, q_max_idx = torch.max(self.q(obs), dim=1)
        return q_max_val.item(), q_max_idx.item()

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

        obs_v = self.convert_to_tensor(obs, dtype=torch.float32)
        actions_v = self.convert_to_tensor(actions, dtype=torch.int64)
        rewards_v = self.convert_to_tensor(rewards, dtype=torch.float32)
        terminations_v = self.convert_to_tensor(terminations, dtype=torch.bool)
        truncations_v = self.convert_to_tensor(truncations, dtype=torch.bool)
        next_obs_v = self.convert_to_tensor(next_obs, dtype=torch.float32)

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
        if len(self.exp_buffer) < self.replay_size:
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
