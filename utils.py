import gym
import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np
import torch
from torch.distributions import Categorical
from collections import namedtuple
from gym_minigrid.wrappers import *

### Environment Fucntions ###

class CardinalActionWrapper(gym.Wrapper):
    """
    Changes the actions to be cardinal directions
    from turning and moving forward.
    """

    # Mapping from action idx to target direction
    act_dir_map = {
        0: 3
    }

    def __init__(self, env):
        super().__init__(env)

        # 4 actions: up, down, left, right
        self.action_space = gym.spaces.Discrete(4)
        # Take out the agent direction from the observation
        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 2),  # number of cells
            dtype='uint8'
        )

    def action(self, action):
        # Old actions
        #   0: turns counter-clockwise
        #   1: turns clockwise
        #   2: moves forward
        #   3-6: does nothing
        #
        # New actions
        #   0: right
        #   1: down
        #   2: left
        #   3: up
        if action < 0 or action > 3:
            raise ValueError('Action must be between 0 and 3')

        while self.env.unwrapped.agent_dir != action:
            self.env.unwrapped.step_count -= 1
            self.env.unwrapped.step(0)

        return 2

    def observation(self, obs):
        obs['image'] = obs['image'][:, :, :2]
        return obs

    def step(self, action):
        action = self.action(action)
        next_obs, reward, done, info = self.env.step(action)
        next_obs = self.observation(next_obs)
        return next_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

def make_env(max_steps=100):
    env = gym.make('MiniGrid-FourRooms-v0')
    env.max_steps = max_steps
    # env.env.max_steps = max_steps
    env = FullyObsWrapper(env) # Get pixel observations
    env = CardinalActionWrapper(env) # Change actions to cardinal directions
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    return env

def render_state(env):
    img = env.render(mode='rgb_array')
    plt.imshow(img)
    plt.show()

def preprocess_obs(obs):
    obs = np.array(obs, dtype=np.float32)
    obs /= 10.0
    obs = rearrange(obs, 'h w c -> c h w')
    return obs

def obs_to_tensor(obs, device='cuda'):
    obs = torch.from_numpy(obs).to(device)
    if len(obs.shape) == 3:
        obs = obs.unsqueeze(0)
    elif len(obs.shape) != 4:
        raise ValueError('Invalid obs shape')
    return obs

def reset_env(env, seed=None):
    if seed is not None:
        env.seed(seed)
    return env.reset()


### Rollout Functions ###


def discount_rewards(rewards, gamma=0.99):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])


TrajectoryData = namedtuple(
  'TrajectoryData',
  ['obs',
   'acts',
   'rewards',
   'baselines',
   'act_log_probs'])

def rollout(model, env, max_steps=1000, data_hook=None):
    """
    Performs a single rollout.
    Returns training data in the shape (n_steps, observation_shape)
    and the cumulative reward.
    """
    device = next(model.parameters()).device
    train_data = [[], [], [], [], []] # obs, act, reward, values, act_log_probs
    obs = reset_env(env)
    
    ep_reward = 0
    for _ in range(max_steps):
        logits, val = model(obs_to_tensor(obs, device))
        act_distribution = Categorical(logits=logits)
        act = act_distribution.sample()
        act_log_prob = act_distribution.log_prob(act).item()

        act, val = act.item(), val.item()

        next_obs, reward, done, _ = env.step(act)
        if data_hook is not None:
            obs, act, reward, next_obs, done = \
                data_hook(obs, act, reward, next_obs, done)

        for i, item in enumerate((obs, act, reward, val, act_log_prob)):
          train_data[i].append(item)

        obs = next_obs
        ep_reward += reward

        if done:
            break

    train_data[0].append(obs)
    train_data = [np.asarray(x) for x in train_data]
    # Calculate GAEs and replace values with GAE values.
    train_data[3] = calculate_gaes(train_data[2], train_data[3])
    
    train_data = TrajectoryData(*train_data)

    return train_data, ep_reward