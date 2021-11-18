import torch
from torch import optim
from torch.distributions.categorical import Categorical
import numpy as np
import torch

from utils import *

class PPOTrainer():
  def __init__(self,
               actor_critic,
               ppo_clip_val=0.2,
               target_kl_div=0.01,
               max_policy_train_iters=80,
               value_train_iters=80,
               policy_lr=3e-4,
               value_lr=1e-2):
    self.ac = actor_critic
    self.ppo_clip_val = ppo_clip_val
    self.target_kl_div = target_kl_div
    self.max_policy_train_iters = max_policy_train_iters
    self.value_train_iters = value_train_iters

    policy_params = list(self.ac.shared_layers.parameters()) + \
        list(self.ac.policy_layers.parameters())
    self.policy_optim = optim.Adam(policy_params, lr=policy_lr)

    value_params = list(self.ac.shared_layers.parameters()) + \
        list(self.ac.value_layers.parameters())
    self.value_optim = optim.Adam(value_params, lr=value_lr)

  def train_policy(self, obs, acts, old_log_probs, gaes):
    for _ in range(self.max_policy_train_iters):
      self.policy_optim.zero_grad()

      new_logits = self.ac.policy(obs)
      new_logits = Categorical(logits=new_logits)
      new_log_probs = new_logits.log_prob(acts)
      # new_selected_logits = new_logits.gather(1, acts.unsqueeze(-1))

      policy_ratio = torch.exp(new_log_probs - old_log_probs)
      clipped_ratio = policy_ratio.clamp(
          1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
      
      full_loss = policy_ratio * gaes
      clipped_loss = clipped_ratio * gaes
      policy_loss = -torch.min(full_loss, clipped_loss).mean()

      policy_loss.backward()
      self.policy_optim.step()

      kl_div = (old_log_probs - new_log_probs).mean()
      if kl_div >= self.target_kl_div:
        break

  def train_value(self, obs, returns):
    for _ in range(self.value_train_iters):
      self.value_optim.zero_grad()

      values = self.ac.value(obs)
      value_loss = (returns - values) ** 2
      value_loss = value_loss.mean()

      value_loss.backward()
      self.value_optim.step()


def train_with_ppo(
    model,
    trainer,
    env,
    max_rollout_steps = 1000,
    n_episodes = None,
    n_steps = None,
    print_freq = None,
    data_modifier_hook = None,
    rollout_data_hook = None):

  # Taking the XOR
  assert (n_steps is None) != (n_episodes is None), \
    'Must specify only n_steps or only n_episodes'

  device = next(model.parameters()).device
  ep_rewards = []

  episode_idx = 0
  step_idx = 0
  while True:
    if n_episodes is not None and episode_idx >= n_episodes:
      break
    elif step_idx >= n_steps:
      break

    # Perform rollout
    max_steps = min(max_rollout_steps, n_steps - step_idx)
    train_data, reward = rollout(model, env,
      max_steps=max_steps, data_hook=rollout_data_hook)
    step_idx += len(train_data[0])
    ep_rewards.append(reward)
    
    if data_modifier_hook:
      train_data = data_modifier_hook(train_data)

    # Shuffle data
    permute_idxs = np.random.permutation(len(train_data[1]))

    obs = torch.tensor(train_data[0][:-1][permute_idxs],
                       dtype=torch.float32, device=device)
    acts = torch.tensor(train_data[1][permute_idxs],
                        dtype=torch.int32, device=device)
    gaes = torch.tensor(train_data[3][permute_idxs],
                        dtype=torch.float32, device=device)
    act_log_probs = torch.tensor(train_data[4][permute_idxs],
                                 dtype=torch.float32, device=device)

    returns = discount_rewards(train_data[2])[permute_idxs]
    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    # Train model
    trainer.train_policy(obs, acts, act_log_probs, gaes)
    trainer.train_value(obs, returns)

    if print_freq is not None and (episode_idx + 1) % print_freq == 0:
      print('Episode {} | Step {} | Avg Reward {}'.format(
          episode_idx + 1, step_idx + 1, np.mean(ep_rewards[-print_freq:])))

    episode_idx += 1
      
  return ep_rewards