from collections import defaultdict

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

ENV_NAME = "FrozenLake-v1"

GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

ENV_NAME = "FrozenLake-v1"
MAP_NAME = "4x4"


class Agent:
    def __init__(self):
        super().__init__()
        self.env = gym.make(
            ENV_NAME,
            desc=None,
            map_name=MAP_NAME,
            is_slippery=True,
            render_mode="rgb_array",
        )
        self.state, _ = self.env.reset()
        # Note that we don't need to track the history of rewards and transitions
        # compared to the value iteration agent
        self.values = defaultdict(float)

    def sample_env(self):
        a = self.env.action_space.sample()
        s = self.state
        s_prime, r, terminated, truncated, _ = self.env.step(a)
        if terminated or truncated:
            self.state, _ = self.env.reset()
        else:
            self.state = s_prime
        return s, a, r, s_prime

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        action_space = self.env.action_space.n
        for action in range(action_space):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, s_prime):
        old_v = self.values[(s, a)]
        best_v, _ = self.best_value_and_action(s_prime)
        new_v = r + GAMMA * best_v
        self.values[(s, a)] = (1 - ALPHA) * old_v + ALPHA * new_v

    def play_episode(self, env, record=False):
        if record is True:
            # Add video recording
            # Requires render_mode="rgb_array"
            # !! Requires ffmpeg (e.g. `brew install ffmpeg`)
            env = gym.wrappers.RecordVideo(env, "recording")
        state, _ = env.reset()
        total_reward = 0.0
        while True:
            _, action = self.best_value_and_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            else:
                state = next_state

        return total_reward


if __name__ == "__main__":
    test_env = gym.make(
        ENV_NAME,
        desc=None,
        map_name=MAP_NAME,
        is_slippery=True,
        render_mode="rgb_array",
    )
    agent = Agent()

    writer = SummaryWriter(comment="-tabular-q-learning")

    iter_no = 0.0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, s_prime = agent.sample_env()
        agent.value_update(s, a, r, s_prime)

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

    print("Q(s,a) Table")
    print(agent.values)

    writer.close()
