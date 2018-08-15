
import numpy as np

class ActionRolloutStorage(object):
    def __init__(self, memory_capacity):
        self.memory_capacity = memory_capacity
        self.reset()

    def insert(self, action, action_log_prob, value, reward, reset):
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.resets.append(reset)
        self.step += 1

    def _compute_returns(self, gamma):
        returns = []
        reward_sum = 0.0
        for reward, reset in zip(self.rewards[::-1], self.resets[::-1]):
            if reset: 
                reward_sum = 0.0
            reward_sum = reward + gamma * reward_sum
            returns.append(reward_sum)
        returns.reverse()
        return returns

    def _compute_advantages(self, returns, z_scale=True):
        advantages = returns - self.values
        if z_scale:
            advantages_mean = np.mean(advantages)
            advantages_stddev = np.std(advantages)
            advantages -= advantages_mean
            advantages /= (advantages_stddev + 1e-8)
        return advantages

    def batch(self, gamma):
        returns = self._compute_returns(gamma)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        returns = np.array(returns)
        action_log_probs = np.array(self.action_log_probs)
        actions = np.array(self.actions)
        advantages = self._compute_advantages(returns, z_scale=True)
        # print('rw|rt|a')
        # for rw, rt, a in zip(rewards, returns, advantages):
        #     print('{}|{}|{}'.format(rw, rt, a))
        # input()
        return rewards, values, returns, action_log_probs, actions, advantages

    def reset(self):
        self.rewards = []
        self.values = []
        self.action_log_probs = []
        self.actions = []
        self.resets = []
        self.step = 0