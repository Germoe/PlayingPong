import argparse
import os
import random
import sys
from datetime import datetime

import marl_pong_env
import torch
from marl_agent import DQNAgent

TEST_SIZE = 1000


class MARL_Testing:
    def __init__(self, env, device, dimensions, test_size=TEST_SIZE):
        self.env = env
        self.frame = 0
        self.test_size = test_size
        self.device = device
        self.sampling_rate = 0.01
        self.test_set = torch.empty((0, *dimensions), device=device)

        self._reset()

    def init_test_set(self):
        """
        Tensor of observations for the test set. Use a torch
        """
        print(
            f"Expected number of frames required for sampling: "
            f"{int(self.test_size / self.sampling_rate)} frames"
        )

        while True:
            obs, _, _, _, _ = self.env.last()
            if random.random() < self.sampling_rate:
                obs_v = torch.tensor(obs, dtype=torch.float32, device=self.device)
                self.test_set = torch.cat((self.test_set, obs_v.unsqueeze(0)))

            if len(self.test_set) >= self.test_size:
                break

            if self.frame % 5000 == 0:
                sys.stdout.write(
                    f"\rNumber of frames sampled: {len(self.test_set)}/{self.test_size}"
                )
                sys.stdout.flush()
            self.random_step()

    def _reset(self):
        self.env.reset()
        self.agent_iter = iter(self.env.agent_iter())

    def random_step(self):
        """
        The step function is the main env sampling function for the
        algorithm. It is used to generate a test set for the algorithm.
        """

        try:
            a_id = next(self.agent_iter)
        except StopIteration:
            self._reset()
            return None

        _, _, terminated, truncated, _ = self.env.last()

        if terminated or truncated:
            action = None
        else:
            action = self.env.action_space(a_id).sample()

        self.env.step(action)
        self.frame += 1

        return a_id

    def save(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(
            self.test_set,
            f"{dir}/test_set_{datetime.now().isoformat()[:-7].replace(':','_')}.pt",
        )
        return f"{dir}/test_set_{datetime.now().isoformat()[:-7].replace(':','_')}.pt"

    def load(self, path):
        self.test_set = torch.load(path).to(self.device)
        print(f"Loaded test set from {path}")
        print(f"Test set size: {self.test_set.shape}")

    @torch.no_grad()
    def run_test(self, agent):
        """
        Pass the test set through the agent and return the average Q value
        """
        out_v = agent.net(self.test_set)
        q_vals_v, _ = torch.max(out_v, dim=1)
        return torch.mean(q_vals_v).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init",
        default=False,
        action="store_true",
        help="Initialize a test set",
    )
    parser.add_argument(
        "--load",
        default=False,
        type=str,
        help="Load a test set",
    )
    parser.add_argument(
        "--agents",
        default=False,
        type=str,
        nargs="+",
        help="Load DQN from file",
    )

    args = parser.parse_args()
    env = marl_pong_env.make_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    print("Observation space: ", env.observation_space(env.possible_agents[0]).shape)

    MARL_test = MARL_Testing(
        env,
        dimensions=env.observation_space(env.possible_agents[0]).shape,
        device=device,
    )

    if args.init is True:
        MARL_test.init_test_set()

        # Save the test set
        MARL_test.save(
            f"tests/test_set_{datetime.now().isoformat()[:-7]}.pt".replace(":", "_")
        )

    else:
        if args.load is False:
            raise Exception("No test set specified")
        if args.agents is False:
            raise Exception("No agents specified")
        if len(args.agents) != len(env.possible_agents):
            raise Exception(
                f"Number of agents ({len(args.agents)}) does not match number of"
                f" agents in the environment ({len(env.possible_agents)})"
            )

        # Load the test set
        MARL_test.load(args.load)

        # Load the agents
        agents = {}
        for i, agent_id in enumerate(env.possible_agents):
            agents[agent_id] = DQNAgent(
                env,
                agent_id,
                epsilon_start=0.05,
                epsilon_end=0.05,
                epsilon_decay_last_frame=1,
                device=device,
                load=args.agents[i],
            )

        for agent in agents.values():
            print(MARL_test.run_test(agent))
