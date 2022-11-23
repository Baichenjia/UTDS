import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
import random
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import json
from pathlib import Path
import hydra
import numpy as np
import torch
from dm_env import specs
import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader
from video import VideoRecorder
import wandb
from omegaconf import OmegaConf


torch.backends.cudnn.benchmark = True

with open("task.json", "r") as f:
	task_dict = json.load(f)


def get_domain(task):
	if task.startswith('point_mass_maze'):
		return 'point_mass_maze'
	return task.split('_', 1)[0]


def get_data_seed(seed, num_data_seeds):
	return (seed - 1) % num_data_seeds + 1


def eval(global_step, agent, env, logger, num_eval_episodes, video_recorder):
	step, episode, total_reward = 0, 0, 0
	eval_until_episode = utils.Until(num_eval_episodes)
	while eval_until_episode(episode):
		time_step = env.reset()
		video_recorder.init(env, enabled=(episode == 0))
		while not time_step.last():
			with torch.no_grad(), utils.eval_mode(agent):
				action = agent.act(time_step.observation, step=global_step, eval_mode=True)
			time_step = env.step(action)
			video_recorder.record(env)
			total_reward += time_step.reward
			step += 1

		episode += 1
		video_recorder.save(f'{global_step}.mp4')

	with logger.log_and_dump_ctx(global_step, ty='eval') as log:
		log('episode_reward', total_reward / episode)
		log('episode_length', step / episode)
		log('step', global_step)
	return {"eval_episode_reward": total_reward / episode}


@hydra.main(config_path='.', config_name='config_single')
def main(cfg):
	work_dir = Path.cwd()
	print(f'workspace: {work_dir}')

	# random seeds
	cfg.seed = random.randint(0, 100000)

	utils.set_seed_everywhere(cfg.seed)
	device = torch.device(cfg.device)

	# create logger
	logger = Logger(work_dir, use_tb=cfg.use_tb)

	# create envs
	env = dmc.make(cfg.task, seed=cfg.seed)

	# create agent
	agent = hydra.utils.instantiate(cfg.agent,
		obs_shape=env.observation_spec().shape, action_shape=env.action_spec().shape,
	    num_expl_steps=0)

	# create replay buffer
	replay_dir_list = []
	datasets_dir = work_dir / cfg.replay_buffer_dir
	replay_dir = datasets_dir.resolve() / Path(cfg.task+"-td3-"+str(cfg.data_main)) / 'data'
	print(f'replay dir: {replay_dir}')
	replay_dir_list.append(replay_dir)

	# 构建 replay buffer (single task)
	replay_loader = make_replay_loader(env, replay_dir_list, cfg.replay_buffer_size,
				cfg.batch_size, cfg.replay_buffer_num_workers, cfg.discount,
				main_task=cfg.task, task_list=[cfg.task])
	replay_iter = iter(replay_loader)     # OfflineReplayBuffer.sample function
	print("load data done.")

	# create video recorders
	video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

	timer = utils.Timer()
	global_step = 0

	train_until_step = utils.Until(cfg.num_grad_steps)
	eval_every_step = utils.Every(cfg.eval_every_steps)
	log_every_step = utils.Every(cfg.log_every_steps)

	if cfg.wandb:
		wandb_dir = f"./wandb/{cfg.task}_{cfg.agent.name}_{cfg.data_main}_{cfg.seed}"
		if not os.path.exists(wandb_dir):
			os.makedirs(wandb_dir)
		wandb.init(project="UTDS", entity='', config=cfg, group=f'{cfg.task}_{cfg.agent.name}_{cfg.data_main}',
		           name=f'{cfg.task}_{cfg.agent.name}_{cfg.data_main}', dir=wandb_dir)
		wandb.config.update(vars(cfg))

	while train_until_step(global_step):
		# try to evaluate
		if eval_every_step(global_step):
			logger.log('eval_total_time', timer.total_time(), global_step)
			eval_metrics = eval(global_step, agent, env, logger, cfg.num_eval_episodes, video_recorder)
			if cfg.wandb:
				wandb.log(eval_metrics)

		# train the agent
		metrics = agent.update(replay_iter, global_step, cfg.num_grad_steps)
		if cfg.wandb:
			wandb.log(metrics)

		# log
		logger.log_metrics(metrics, global_step, ty='train')
		if log_every_step(global_step):
			elapsed_time, total_time = timer.reset()
			with logger.log_and_dump_ctx(global_step, ty='train') as log:
				log('fps', cfg.log_every_steps / elapsed_time)
				log('total_time', total_time)
				log('step', global_step)

		global_step += 1


if __name__ == '__main__':
	main()
