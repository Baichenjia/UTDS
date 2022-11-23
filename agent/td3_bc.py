import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import copy
import utils
from dm_control.utils import rewards


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action=1):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)

		self.max_action = max_action

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3BCAgent:
	def __init__(self,
				 name,
				 obs_shape,
				 action_shape,
				 device,
				 lr,
				 hidden_dim,
				 critic_target_tau,
				 actor_target_tau,
				 policy_freq,
	             policy_noise,
	             noise_clip,
				 use_tb,
				 alpha,
				 batch_size,
				 num_expl_steps):
		self.policy_noise = policy_noise
		self.policy_freq = policy_freq
		self.noise_clip = noise_clip
		self.num_expl_steps = num_expl_steps
		self.action_dim = action_shape[0]
		self.hidden_dim = hidden_dim
		self.lr = lr
		self.device = device
		self.critic_target_tau = critic_target_tau
		self.actor_target_tau = actor_target_tau
		self.use_tb = use_tb
		# self.stddev_schedule = stddev_schedule
		# self.stddev_clip = stddev_clip
		self.alpha = alpha
		self.max_action = 1.0

		# models
		self.actor = Actor(obs_shape[0], action_shape[0]).to(device)
		self.actor_target = copy.deepcopy(self.actor)

		self.critic = Critic(obs_shape[0], action_shape[0]).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_target.load_state_dict(self.critic.state_dict())

		# optimizers
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.train()
		self.critic_target.train()

	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		self.critic.train(training)

	def act(self, obs, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
		action = self.actor(obs)
		if step < self.num_expl_steps:
			action.uniform_(-1.0, 1.0)
		return action.cpu().numpy()[0]

	def update_critic(self, obs, action, reward, discount, next_obs):
		metrics = dict()

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
					torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = (
					self.actor_target(next_obs) + noise
			).clamp(-self.max_action, self.max_action)

			target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
			target_V = torch.min(target_Q1, target_Q2)
			target_Q = reward + (discount * target_V)

		Q1, Q2 = self.critic(obs, action)
		critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

		if self.use_tb:
			metrics['critic_target_q'] = target_Q.mean().item()
			metrics['critic_q1'] = Q1.mean().item()
			metrics['critic_q2'] = Q2.mean().item()
			metrics['critic_loss'] = critic_loss.item()

		# optimize critic
		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_opt.step()
		return metrics

	def update_actor(self, obs, action):
		metrics = dict()

		# Compute actor loss
		pi = self.actor(obs)
		Q = self.critic.Q1(obs, pi)
		lmbda = self.alpha / Q.abs().mean().detach()

		actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()

		return metrics

	def update(self, replay_iter, step, total_step=None):
		metrics = dict()

		batch = next(replay_iter)
		# obs.shape=(1024,obs_dim), action.shape=(1024,1), reward.shape=(1024,1), discount.shape=(1024,1)
		obs, action, reward, discount, next_obs, _ = utils.to_torch(
			batch, self.device)

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()

		# update critic
		metrics.update(
			self.update_critic(obs, action, reward, discount, next_obs))

		# update actor
		if step % self.policy_freq == 0:
			metrics.update(self.update_actor(obs, action))

		# update critic target
		utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
		utils.soft_update_params(self.actor, self.actor_target, self.actor_target_tau)

		return metrics
