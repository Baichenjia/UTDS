import warnings
import wandb
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import pickle
from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer_collect import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, num_expl_steps, cfg):
	cfg.obs_shape = obs_spec.shape
	cfg.action_shape = action_spec.shape
	cfg.num_expl_steps = num_expl_steps
	return hydra.utils.instantiate(cfg)


class Workspace:
	def __init__(self, cfg):
		self.work_dir = Path.cwd()
		print(f'workspace: {self.work_dir}')

		self.cfg = cfg
		utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)

		# create logger
		self.logger = Logger(self.work_dir, use_tb=cfg.use_tb)
		self.train_env = dmc.make(cfg.task, action_repeat=cfg.action_repeat, seed=cfg.seed)
		self.eval_env = dmc.make(cfg.task, action_repeat=cfg.action_repeat, seed=cfg.seed)

		# create agent
		self.agent = make_agent(self.train_env.observation_spec(),
								self.train_env.action_spec(),
								cfg.num_seed_frames // cfg.action_repeat,
								cfg.agent)

		# get meta specs
		meta_specs = self.agent.get_meta_specs()
		# create replay buffer
		data_specs = (self.train_env.observation_spec(), self.train_env.action_spec(),
					  specs.Array((1,), np.float32, 'reward'),
					  specs.Array((1,), np.float32, 'discount'))

		# create data storage
		self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
		                replay_dir=self.work_dir / 'buffer', dataset_dir=self.work_dir / 'data')

		# create replay buffer
		self.replay_loader = make_replay_loader(self.replay_storage, cfg.replay_buffer_size,
					cfg.batch_size, cfg.replay_buffer_num_workers, False, cfg.nstep, cfg.discount)
		self._replay_iter = None

		# create video recorders
		self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
		self.train_video_recorder = TrainVideoRecorder(self.work_dir if cfg.save_train_video else None)

		self.timer = utils.Timer()
		self._global_step = 0
		self._global_episode = 0

		# TODO: save agent
		self._agent_dir = self.work_dir / 'agent'
		self._agent_dir.mkdir(exist_ok=True)
		self.change_freq = True

	@property
	def global_step(self):
		return self._global_step

	@property
	def global_episode(self):
		return self._global_episode

	@property
	def global_frame(self):
		return self.global_step * self.cfg.action_repeat

	@property
	def replay_iter(self):
		if self._replay_iter is None:
			self._replay_iter = iter(self.replay_loader)
		return self._replay_iter

	def eval(self):
		step, episode, total_reward = 0, 0, 0
		eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
		meta = self.agent.init_meta()
		while eval_until_episode(episode):      # eval 10 episodes
			time_step = self.eval_env.reset()
			self.video_recorder.init(self.eval_env, enabled=(episode == 0))
			while not time_step.last():
				with torch.no_grad(), utils.eval_mode(self.agent):
					action = self.agent.act(time_step.observation, step=self.global_step, eval_mode=True, meta=meta)
				time_step = self.eval_env.step(action)
				self.video_recorder.record(self.eval_env)
				total_reward += time_step.reward
				step += 1
			episode += 1
			self.video_recorder.save(f'{self.global_frame}.mp4')

		with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
			log('episode_reward', total_reward / episode)
			log('episode_length', step * self.cfg.action_repeat / episode)
			log('episode', self.global_episode)
			log('step', self.global_step)
		if self.cfg.use_wandb:
			wandb.log({"eval_return": total_reward / episode})

		return total_reward / episode

	def train(self):
		train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
		seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
		eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)
		save_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)  # TODO: save agent

		episode_step, episode_reward = 0, 0
		time_step = self.train_env.reset()
		meta = self.agent.init_meta()
		self.replay_storage.add(time_step, meta, physics=self.train_env.physics.get_state())   # 这里加入了physics信息
		self.train_video_recorder.init(time_step.observation)
		metrics = None
		eval_rew = 0
		while train_until_step(self.global_step):
			if time_step.last():
				self._global_episode += 1
				self.train_video_recorder.save(f'{self.global_frame}.mp4')
				# wait until all the metrics schema is populated
				if metrics is not None:
					# log stats
					elapsed_time, total_time = self.timer.reset()
					episode_frame = episode_step * self.cfg.action_repeat
					with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
						log('fps', episode_frame / elapsed_time)
						log('total_time', total_time)
						log('episode_reward', episode_reward)
						log('episode_length', episode_frame)
						log('episode', self.global_episode)
						log('buffer_size', len(self.replay_storage))
						log('step', self.global_step)

				# reset env
				time_step = self.train_env.reset()
				meta = self.agent.init_meta()
				self.replay_storage.add(time_step, meta, physics=self.train_env.physics.get_state())
				self.train_video_recorder.init(time_step.observation)

				episode_step = 0
				episode_reward = 0

			# try to evaluate
			if eval_every_step(self.global_step):
				self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
				eval_rew = self.eval()

			# TODO: save policy
			if save_every_step(self.global_step):
				agent_stamp = self._agent_dir / f'agent-{int(self.global_step/1000)}K-{round(eval_rew, 2)}.pkl'
				with open(str(agent_stamp), 'wb') as f_agent:
					pickle.dump(self.agent, f_agent)
				print("Save agent to", agent_stamp)
				if self.global_step >= 200000 and self.change_freq:
					save_every_step.change_every(freq=10)       # decrease the freq
					self.change_freq = False

			meta = self.agent.update_meta(meta, self.global_step, time_step)
			if hasattr(self.agent, "regress_meta"):
				repeat = self.cfg.action_repeat
				every = self.agent.update_task_every_step // repeat
				init_step = self.agent.num_init_steps
				if self.global_step > (init_step // repeat) and self.global_step % every == 0:
					meta = self.agent.regress_meta(self.replay_iter, self.global_step)

			# sample action
			with torch.no_grad(), utils.eval_mode(self.agent):
				action = self.agent.act(time_step.observation,
				            meta=meta, step=self.global_step, eval_mode=False)

			# try to update the agent
			if not seed_until_step(self.global_step):
				metrics = self.agent.update(self.replay_iter, self.global_step)
				self.logger.log_metrics(metrics, self.global_frame, ty='train')
				if self.cfg.use_wandb:
					wandb.log(metrics)

			# take env step
			time_step = self.train_env.step(action)
			episode_reward += time_step.reward
			self.replay_storage.add(time_step, meta, physics=self.train_env.physics.get_state())
			self.train_video_recorder.record(time_step.observation)
			episode_step += 1
			self._global_step += 1


@hydra.main(config_path='.', config_name='collect_data')
def main(cfg):
	from collect_data import Workspace as W
	root_dir = Path.cwd()
	workspace = W(cfg)
	snapshot = root_dir / 'snapshot.pt'
	if snapshot.exists():
		print(f'resuming: {snapshot}')
		workspace.load_snapshot()

	if cfg.use_wandb:
		wandb_dir = f"./wandb/collect_{cfg.task}_{cfg.agent.name}_{cfg.seed}"
		if not os.path.exists(wandb_dir):
			os.makedirs(wandb_dir)
		wandb.init(project="UTDS", entity='', config=cfg, group=f'{cfg.task}_{cfg.agent.name}',
		           name=f'{cfg.task}_{cfg.agent.name}', dir=wandb_dir)
		wandb.config.update(vars(cfg))

	workspace.train()


if __name__ == '__main__':
	main()
