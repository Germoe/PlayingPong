from collections import Counter, defaultdict

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

ENV_NAME = "FrozenLake-v1"

GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        super().__init__()
        self.env = gym.make(
            ENV_NAME,
            desc=None,
            map_name="8x8",
            is_slippery=True,
            render_mode="rgb_array",
        )
        self.state, _ = self.env.reset()
        self.reward = defaultdict(float)
        self.transitions = defaultdict(Counter)
        self.values = defaultdict(float)

    def play_n_random_steps(self, n):
        for _ in range(n):
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.reward[(self.state, action, next_state)] = reward
            self.transitions[(self.state, action)][next_state] += 1
            if terminated or truncated:
                self.state, _ = self.env.reset()
            else:
                self.state = next_state

    def calc_action_value(self, state, action):
        target_counts = self.transitions[(state, action)]
        total = sum(target_counts.values())
        q_sa = 0

        for target_s, count in target_counts.items():
            reward = self.reward[(state, action, target_s)]
            q_sa += count / total * (reward + GAMMA * self.values[target_s])
        return q_sa

    def select_action(self, state):
        best_action, best_value = None, None
        action_space = self.env.action_space.n
        for action in range(action_space):
            q_sa = self.calc_action_value(state, action)
            if best_value is None or best_value < q_sa:
                best_value = q_sa
                best_action = action
        return best_action

    def play_episode(self, env, record=False):
        if record is True:
            # Add video recording
            # Requires render_mode="rgb_array"
            # !! Requires ffmpeg (e.g. `brew install ffmpeg`)
            env = gym.wrappers.RecordVideo(env, "recording")
        state, _ = env.reset()
        total_reward = 0.0
        while True:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            self.reward[(state, action, next_state)] = reward
            self.transitions[(state, action)][next_state] += 1
            total_reward += reward
            if terminated or truncated:
                break
            else:
                state = next_state

        return total_reward

    def value_iteration(self):
        action_space = self.env.action_space.n
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calc_action_value(state, action) for action in range(action_space)
            ]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="8x8",
        is_slippery=True,
        render_mode="rgb_array",
    )
    agent = Agent()

    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0.0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward_mean", reward, iter_no)
        if reward > best_reward:
            print(f"Best Reward updated {best_reward} -> {reward}")
            best_reward = reward
        if reward > 0.8:
            print(f"Solved in {iter_no} iterations!")
            break

    # Record Final try
    agent.play_episode(test_env, record=True)

    print("Reward Table")
    print(agent.reward)

    print("V(s) Table")
    print(agent.values)

    print("Transition Table")
    print(agent.transitions)

    writer.close()
