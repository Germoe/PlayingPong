import argparse
import datetime

import marl_pong_env
import torch
from marl_training import DQNAgent, MARL_DQN_Algorithm

EPSILON_START = 0
EPSILON_END = 0
EPSILON_DECAY_LAST_FRAME = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable CUDA"
    )
    parser.add_argument(
        "--record",
        default=False,
        action="store_true",
        help="Record a video of the agent playing",
    )
    parser.add_argument(
        "--agent0",
        default=False,
        type=str,
        help="Load agent 0 from file",
    )
    parser.add_argument(
        "--agent1",
        default=False,
        type=str,
        help="Load agent 1 from file",
    )

    args = parser.parse_args()
    agents_to_load = [args.agent0, args.agent1]
    env = marl_pong_env.make_env(args.record)
    device = torch.device("cuda" if args.cuda else "cpu")

    run_name = f"pong-marl-{datetime.datetime.now().isoformat()}".replace(":", "_")
    print("Using device: ", device)

    agents = {}
    for i, agent_id in enumerate(env.possible_agents):
        agents[agent_id] = DQNAgent(
            env,
            agent_id,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay_last_frame=EPSILON_DECAY_LAST_FRAME,
            device=device,
            load=agents_to_load[i],
        )

    n_agents = len(agents)

    marl = MARL_DQN_Algorithm(env, agents)
    train_agents = {agent_id: False for agent_id in env.possible_agents}
    marl.set_train_agents(train_agents)

    while True:
        a_id = marl.step()
        if a_id is None:
            # Print the rewards for each agent
            for agent_id in env.possible_agents:
                print(f"Agent {agent_id} reward: {marl.agents[agent_id].rewards}")
            break
