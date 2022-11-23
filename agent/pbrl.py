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
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class PBRLAgent:
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
				 # alpha,
				 batch_size,
				 num_expl_steps,
	             # PBRL parameters
	             num_random,
	             ucb_ratio_in,
	             ucb_ratio_ood_init,
	             ucb_ratio_ood_min,
	             ood_decay_factor,
	             ensemble,
	             ood_noise,
	             share_ratio,
	             has_next_action=False):
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
		# self.alpha = alpha
		self.max_action = 1.0
		self.share_ratio = share_ratio
		self.share_ratio_now = None

		# PBRL parameters
		self.num_random = num_random      # for ood action
		self.ucb_ratio_in = ucb_ratio_in
		self.ensemble = ensemble
		self.ood_noise = ood_noise

		# PBRL parameters: control ood ratio
		self.ucb_ratio_ood_init = ucb_ratio_ood_init
		self.ucb_ratio_ood_min = ucb_ratio_ood_min
		self.ood_decay_factor = ood_decay_factor
		self.ucb_ratio_ood = ucb_ratio_ood_init
		self.ucb_ratio_ood_linear_steps = None

		# models
		self.actor = Actor(obs_shape[0], action_shape[0]).to(device)
		self.actor_target = copy.deepcopy(self.actor)

		# initialize ensemble of critic
		self.critic, self.critic_target = [], []
		for _ in range(self.ensemble):
			single_critic = Critic(obs_shape[0], action_shape[0]).to(device)
			single_critic_target = copy.deepcopy(single_critic)
			single_critic_target.load_state_dict(single_critic.state_dict())
			self.critic.append(single_critic)
			self.critic_target.append(single_critic_target)
		print("Actor parameters:", utils.total_parameters(self.actor))
		print("Critic parameters: single", utils.total_parameters(self.critic[0]), ", total:", utils.total_parameters(self.critic))

		# optimizers
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_opt = []
		for i in range(self.ensemble):      # each ensemble member has its optimizer
			self.critic_opt.append(torch.optim.Adam(self.critic[i].parameters(), lr=lr))

		self.train()
		for ct_single in self.critic_target:
			ct_single.train()

	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		for c_single in self.critic:
			c_single.train(training)

	def act(self, obs, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
		action = self.actor(obs)
		if step < self.num_expl_steps:
			action.uniform_(-1.0, 1.0)
		return action.cpu().numpy()[0]

	def ucb_func(self, obs, action, mean=False):
		action_shape = action.shape[0]               # 1024*num_random
		obs_shape = obs.shape[0]                     # 1024
		assert int(action_shape / obs_shape) in [1, self.num_random]
		if int(action_shape / obs_shape) != 1:
			obs = obs.unsqueeze(1).repeat(1, self.num_random, 1).view(obs.shape[0] * self.num_random, obs.shape[1])
		# Bootstrapped uncertainty
		q_pred = []
		for i in range(self.ensemble):
			q_pred.append(self.critic[i](obs.cuda(), action.cuda()))
		ucb = torch.std(torch.hstack(q_pred), dim=1, keepdim=True)   # (1024, ensemble) -> (1024, 1)
		assert ucb.size() == (action_shape, 1)
		if mean:
			q_pred = torch.mean(torch.hstack(q_pred), dim=1, keepdim=True)
		return ucb, q_pred

	def ucb_func_target(self, obs_next, act_next):
		action_shape = act_next.shape[0]             # 2560
		obs_shape = obs_next.shape[0]                # 256
		assert int(action_shape / obs_shape) in [1, self.num_random]
		if int(action_shape / obs_shape) != 1:
			obs_next = obs_next.unsqueeze(1).repeat(1, self.num_random, 1).view(obs_next.shape[0] * self.num_random, obs_next.shape[1])  # （2560, obs_dim）
		# Bootstrapped uncertainty
		target_q_pred = []
		for i in range(self.ensemble):
			target_q_pred.append(self.critic[i](obs_next.cuda(), act_next.cuda()))
		ucb_t = torch.std(torch.hstack(target_q_pred), dim=1, keepdim=True)
		assert ucb_t.size() == (action_shape, 1)
		return ucb_t, target_q_pred

	def update_critic(self, obs, action, reward, discount, next_obs, step, total_step, bool_flag):
		self.share_ratio_now = utils.decay_linear(t=step, init=self.share_ratio, minimum=1.0, total_steps=total_step // 2)

		metrics = dict()
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			next_action = (self.actor_target(next_obs) + noise).clamp(-self.max_action, self.max_action)

			# ood sample 1
			sampled_current_actions = self.actor(obs).unsqueeze(1).repeat(1, self.num_random, 1).view(
						action.shape[0]*self.num_random, action.shape[1])
			noise_current = (torch.randn_like(sampled_current_actions) * self.ood_noise).clamp(-self.noise_clip, self.noise_clip)
			sampled_current_actions = (sampled_current_actions + noise_current).clamp(-self.max_action, self.max_action)

			# ood sample 2
			sampled_next_actions = self.actor(next_obs).unsqueeze(1).repeat(1, self.num_random, 1).view(
						action.shape[0]*self.num_random, action.shape[1])
			noise_next = (torch.randn_like(sampled_next_actions) * self.ood_noise).clamp(-self.noise_clip, self.noise_clip)
			sampled_next_actions = (sampled_next_actions + noise_next).clamp(-self.max_action, self.max_action)

			# random sample
			random_actions = torch.FloatTensor(action.shape[0]*self.num_random, action.shape[1]).uniform_(
						-self.max_action, self.max_action).to(self.device)

		# TODO: UCB and Q-values
		ucb_current, q_pred = self.ucb_func(obs, action)     # (1024,1).  lenth=ensemble, q_pred[0].shape=(1024,1)
		ucb_next, target_q_pred = self.ucb_func_target(next_obs, next_action)   # (1024,1).  lenth=ensemble, target_q_pred[0].shape=(1024,1)

		ucb_curr_actions_ood, qf_curr_actions_all_ood = self.ucb_func(obs, sampled_current_actions)      # (1024*num_random, 1), length=ensemble, (1024*num_random, 1)
		ucb_next_actions_ood, qf_next_actions_all_ood = self.ucb_func(next_obs, sampled_next_actions)    # 同上
		# ucb_rand_ood, qf_rand_actions_all_ood = self.ucb_func(obs, random_actions)

		for qf_index in np.arange(self.ensemble):
			ucb_ratio_in_flag = bool_flag * self.ucb_ratio_in + (1 - bool_flag) * self.ucb_ratio_in * self.share_ratio_now
			ucb_ratio_in_flag = np.expand_dims(ucb_ratio_in_flag, 1)
			q_target = reward + discount * (target_q_pred[qf_index] - torch.from_numpy(ucb_ratio_in_flag.astype(np.float32)).cuda() * ucb_next)  # (1024, 1), (1024, 1), (1024, 1)
			# print("bool flag", bool_flag[:10],  bool_flag[-10:])
			# print("ucb_ratio_in_flag", q_target.shape, ucb_ratio_in_flag.shape, ucb_next.shape, (torch.from_numpy(ucb_ratio_in_flag.astype(np.float32)).cuda() * ucb_next).shape, ucb_ratio_in_flag[:10])

			# q_target = reward + discount * (target_q_pred[qf_index] - self.ucb_ratio_in * ucb_next)  # (1024, 1), (1024, 1), (1024, 1)
			q_target = q_target.detach()
			qf_loss_in = F.mse_loss(q_pred[qf_index], q_target)

			# TODO: ood loss
			cat_qf_ood = torch.cat([qf_curr_actions_all_ood[qf_index],
									qf_next_actions_all_ood[qf_index]], 0)
			# assert cat_qf_ood.size() == (1024*self.num_random*3, 1)

			ucb_ratio_ood_flag = bool_flag * self.ucb_ratio_ood + (1 - bool_flag) * self.ucb_ratio_ood * self.share_ratio_now
			ucb_ratio_ood_flag = np.expand_dims(ucb_ratio_ood_flag, 1).repeat(self.num_random, axis=1).reshape(-1, 1).astype(np.float32)
			# print("ucb_ratio_ood_flag 1", ucb_ratio_ood_flag.shape, ucb_curr_actions_ood.shape)

			cat_qf_ood_target = torch.cat([
				torch.maximum(qf_curr_actions_all_ood[qf_index] - torch.from_numpy(ucb_ratio_ood_flag).cuda() * ucb_curr_actions_ood, torch.zeros(1).cuda()),
				torch.maximum(qf_next_actions_all_ood[qf_index] - torch.from_numpy(ucb_ratio_ood_flag).cuda() * ucb_next_actions_ood, torch.zeros(1).cuda())], 0)
			# print("ucb_ratio_ood_flag 2", cat_qf_ood_target.shape, qf_curr_actions_all_ood[qf_index].shape)
			cat_qf_ood_target = cat_qf_ood_target.detach()

			# assert cat_qf_ood_target.size() == (1024*self.num_random*3, 1)
			qf_loss_ood = F.mse_loss(cat_qf_ood, cat_qf_ood_target)
			critic_loss = qf_loss_in + qf_loss_ood

			# Update the Q-functions
			self.critic_opt[qf_index].zero_grad()
			critic_loss.backward(retain_graph=True)
			self.critic_opt[qf_index].step()

		# change the ood ratio
		self.ucb_ratio_ood = max(self.ucb_ratio_ood_init * self.ood_decay_factor ** step, self.ucb_ratio_ood_min)

		if self.use_tb:
			metrics['critic_target_q'] = q_target.mean().item()
			metrics['critic_q1'] = q_pred[0].mean().item()
			# metrics['critic_q2'] = q_pred[1].mean().item()
			# ucb
			metrics['ucb_current'] = ucb_current.mean().item()
			metrics['ucb_next'] = ucb_next.mean().item()
			metrics['ucb_curr_actions_ood'] = ucb_curr_actions_ood.mean().item()
			metrics['ucb_next_actions_ood'] = ucb_next_actions_ood.mean().item()
			# loss
			metrics['critic_loss_in'] = qf_loss_in.item()
			metrics['critic_loss_ood'] = qf_loss_ood.item()
			metrics['ucb_ratio_ood'] = self.ucb_ratio_ood
			metrics['share_ratio_now'] = self.share_ratio_now
		return metrics

	def update_actor(self, obs, action):
		metrics = dict()

		# Compute actor loss
		pi = self.actor(obs)

		Qvalues = []
		for i in range(self.ensemble):
			Qvalues.append(self.critic[i](obs, pi))           # (1024, 1)
		Qvalues_min = torch.min(torch.hstack(Qvalues), dim=1, keepdim=True).values
		assert Qvalues_min.size() == (1024, 1)

		actor_loss = -1. * Qvalues_min.mean()

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()

		return metrics

	def update(self, replay_iter, step, total_step):
		metrics = dict()

		batch = next(replay_iter)
		obs, action, reward, discount, next_obs, bool_flag = utils.to_torch(
			batch, self.device)
		bool_flag = bool_flag.cpu().detach().numpy()

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()

		# update critic
		metrics.update(
			self.update_critic(obs, action, reward, discount, next_obs, step, total_step, bool_flag))

		# update actor
		if step % self.policy_freq == 0:
			metrics.update(self.update_actor(obs, action))

		# update actor target
		utils.soft_update_params(self.actor, self.actor_target, self.actor_target_tau)

		# update critic target
		for i in range(self.ensemble):
			utils.soft_update_params(self.critic[i], self.critic_target[i], self.critic_target_tau)

		return metrics
