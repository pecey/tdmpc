import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gymnasium as gym
gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import logger
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video, use_policy, use_val_backup):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(num_episodes):
		obs, done, ep_reward, t = env.reset(seed=42000), False, 0, 0
		if video: video.init(env, enabled=(i==0))
		while not done:
			action = agent.plan(obs, eval_mode=True, step=step, t0=t==0, use_policy=use_policy, use_val_backup=use_val_backup)
			obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
			done = terminated or truncated
			ep_reward += reward
			if video: video.record(env)
			t += 1
		episode_rewards.append(ep_reward)
		if video: video.save(env_step)
	return np.nanmean(episode_rewards)


def train(cfg):
	"""Training script for TD-MPC. Requires a CUDA-enabled device."""
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)
	
	# Run training
	L = logger.Logger(work_dir, cfg)
	
	print(f"Working directory: {work_dir}")
	print(f"Config: {cfg}")
	
	episode_idx, start_time = 0, time.time()
	# for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):

	use_policy = cfg.use_policy
	use_val_backup = cfg.use_val_backup

	training = True
	step, env_step = 0, 0
	while episode_idx < 500:
		# Collect trajectory
		obs = env.reset(seed = cfg.seed + episode_idx)
		episode = Episode(cfg, obs)
		while not episode.done:
			action = agent.plan(obs, step=step, t0=episode.first, use_policy=use_policy, use_val_backup=use_val_backup)
			# action = env.action_space.sample()
			obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
			# print(f"Action: {action}, reward: {reward}")
			done = terminated or truncated
			episode += (obs, action, reward, done)
		# assert len(episode) == cfg.episode_length//cfg.action_repeat
		buffer += episode
		step += len(episode)

		# Update model
		train_metrics = {}
		if step >= cfg.seed_steps:
			num_updates = cfg.seed_steps if step == cfg.seed_steps else len(episode)
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+i))

		# Log training episode
		episode_idx += 1
		decision_step = int(step)
		# Obtain the number of environment steps from TEAWrapper or ActionRepeatWrapper.
		env_step += int(env.env.env._env._env.t)
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'decision_step': decision_step,
			'decision_in_episode': len(episode),
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward}
		train_metrics.update(common_metrics)
		L.log(train_metrics, category='train')

		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:
			common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video, use_policy, use_val_backup)
			L.log(common_metrics, category='eval')

	L.finish(agent)
	print('Training completed successfully')


if __name__ == '__main__':
	train(parse_cfg(Path().cwd() / __CONFIG__))
