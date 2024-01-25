import argparse
import datetime
import os
import sys
import time
from itertools import cycle

import marl_pong_env
import numpy as np
import torch
import torch.utils.tensorboard as tb
from marl_agent import DQNAgent
from marl_test import MARL_Testing

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_LAST_FRAME = 1000000
EPOCH_SIZE = 250000

STATIONARY_ITERATIONS = 250000


class MARL_DQN_Algorithm:
    """
    The MARL_DQN_Algorithm class is the main algorithm class for the
    multi-agent reinforcement learning algorithm.

    Input:
        env: The environment to be used for the algorithm.
        agents: A dictionary of agents to be used for the algorithm.
        stationary: An integer indicating the number of frames the
            the other agent is stationary for.
    Attributes:
        env: The environment to be used for the algorithm.
        agents: A dictionary of agents to be used for the algorithm.
        train_agents: A dictionary of agents to be trained.

    """

    def __init__(self, env, agents):
        self.env = env
        self.agents = agents
        self.frame = 0
        self._reset()

    def _reset(self):
        self.env.reset()
        self.agent_iter = iter(self.env.agent_iter())

    def next_agent(self):
        try:
            a_id = next(self.agent_iter)
            return a_id
        except StopIteration:
            self._reset()
            return None

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
            return None, None

        next_obs, reward, terminated, truncated, _ = self.env.last()

        # Receive Reward the active agent has received since last action
        active_a = self.agents[a_id]
        active_a.update_reward(reward)

        if terminated or truncated:
            active_a.append_reward(active_a.curr_reward)
            active_a.reset_reward()
            action = None
        else:
            action = active_a.select_action(next_obs)

        self.env.step(action)
        self.frame += 1

        return a_id, (next_obs, reward, terminated, truncated, _)


class MARL_Train(MARL_DQN_Algorithm):
    def __init__(self, env, agents, stationary=STATIONARY_ITERATIONS):
        super().__init__(env, agents)
        self.stationary = stationary
        self.agent_rotation = cycle(self.env.possible_agents)
        self.active_agent = next(self.agent_rotation)
        print(f"Active agent: {self.active_agent}")

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

    def switch_active_agent(self):
        self.active_agent = next(self.agent_rotation)

    def train_step(self):
        """
        This method extends the step method of the MARL_DQN_Algorithm class
        by adding the training step for the active agent.
        """

        a_id, exp = self.step()
        if a_id is None or exp is None:
            return None

        next_obs, reward, terminated, truncated, _ = exp

        if self.active_agent == a_id:
            self.update_agent(a_id, reward, next_obs, terminated, truncated)

        if self.frame % self.stationary == 0:
            self.switch_active_agent()
            print(f"Active agent: {self.active_agent}")

        return a_id


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
    parser.add_argument(
        "--test",
        default=False,
        type=str,
        help="Load test set from file",
    )

    args = parser.parse_args()
    epsilon_start, epsilon_end, epsilon_decay_last_frame = args.epsilon
    print("Epsilon: ", epsilon_start, epsilon_end, epsilon_decay_last_frame)
    agents_to_load = {0: args.agent0, 1: args.agent1}
    env = marl_pong_env.make_env(record=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    run_name = f"pong-marl-{datetime.datetime.now().isoformat()}".replace(":", "_")
    print("Using device: ", device)
    print(f"Epoch Size: {EPOCH_SIZE}")

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

    marl = MARL_Train(env, agents, stationary=STATIONARY_ITERATIONS)
    marl_test = MARL_Testing(env, device=device)
    if args.test is False:
        marl_test.init_test_set()
        marl_test.save(dir="tests")
    else:
        marl_test.load(args.test)
    iteration = marl.frame
    episodes = 0
    episode_last_iter_n = 0
    last_iteration = 0
    last_save_iteration = 0
    ts_time = time.time()
    n_agents = len(agents)
    mean_reward_10 = {a_id: -100 for a_id in agents}
    mean_reward_100 = {a_id: -100 for a_id in agents}
    test_q_max = {a_id: -100 for a_id in agents}

    writer = tb.SummaryWriter(comment="-pong-marl")

    while True:
        ts_time = time.time()
        last_iteration = iteration

        a_id = marl.train_step()
        if a_id is None:
            episodes += 1
            # Marks the end of an episode

            a_ids = list(marl.env.possible_agents)
            for a_id in a_ids:
                mean_reward_100[a_id] = np.mean(marl.agents[a_id].rewards[-100:])
                mean_reward_10[a_id] = np.mean(marl.agents[a_id].rewards[-10:])
                test_q_max[a_id] = marl_test.run_test(marl.agents[a_id])

                writer.add_scalar(f"avg. q_max/{a_id}", test_q_max[a_id], iteration)

                writer.add_scalar(
                    f"mean-reward-100/{a_id}",
                    mean_reward_100[a_id],
                    iteration,
                )

                writer.add_scalar(
                    f"mean-reward-10/{a_id}",
                    mean_reward_10[a_id],
                    iteration,
                )

            mean_rewards = [f"{mean_reward_100[a_id]:.2f}" for a_id in a_ids]
            mean_rewards_10 = [f"{mean_reward_10[a_id]:.2f}" for a_id in a_ids]

            epsilons = [f"{marl.agents[a_id].epsilon():.3f}" for a_id in a_ids]
            sys.stdout.write(
                f"\rEpisodes: {episodes} | "
                f"Iterations (total): {iteration} |"
                f"Iterations (last): {iteration - episode_last_iter_n} | "
                f"Rw. 100: {mean_rewards} | "
                f"Rw. 10: {mean_rewards_10} | "
                f"Eps: {epsilons}"
            )
            writer.add_scalar(
                "episode_iterations", iteration - episode_last_iter_n, episodes
            )

            episode_last_iter_n = iteration

        # Save model weights at every epoch
        if iteration % EPOCH_SIZE == 0 and episodes > 0:
            # Check if directory exists
            if not os.path.exists(f"model_weights/{run_name}"):
                os.makedirs(f"model_weights/{run_name}")

            for a_id in marl.env.possible_agents:
                torch.save(
                    marl.agents[a_id].net.state_dict(),
                    f"./model_weights/{run_name}/pong-marl-{a_id}-{iteration}.dat",
                )

        # Compute speed f/s
        iteration = marl.frame / n_agents
        speed = (iteration - last_iteration) / (time.time() - ts_time)

        writer.add_scalar(f"epsilon/{a_id}", agents[a_id].epsilon(), iteration)
        writer.add_scalar(f"speed/{a_id}", speed, iteration)
        writer.add_scalar(f"loss/{a_id}", agents[a_id]._loss, iteration)
