from collections import deque, defaultdict
from typing import Any, NamedTuple
import dm_env
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import pixels
from dm_env import StepType, specs
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class ExtendedTimeStep(NamedTuple):
	step_type: Any
	reward: Any
	discount: Any
	observation: Any
	action: Any

	def first(self):
		return self.step_type == StepType.FIRST

	def mid(self):
		return self.step_type == StepType.MID

	def last(self):
		return self.step_type == StepType.LAST

class TemporallyExtendedActionWrapper(dm_env.Environment):
	def __init__(self, env, dt):
		self._env = env
		self._dt = dt

	def step(self, action):
		# Split action to get control and time parts
		ac, t = action[:-1], action[-1]
		reward = 0.0
		discount = 1.0
		# Always perform one step
		repeat = np.maximum((t//self._dt).astype(np.int32), 1)
		# print(f"[TemporallyExtendedActionWrapper] Action : {action}, Repeat: {repeat}")
		for _ in range(repeat):
			self.t += 1
			time_step = self._env.step(ac)
			reward += (time_step.reward or 0.0) * discount
			discount *= time_step.discount
			if time_step.last():
				break

		return time_step._replace(reward=reward, discount=discount)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def reset(self, seed=None, options=None):
		# print(type(self._env))
		self.t = 0
		return self._env.reset(seed=seed, options=options)

	def __getattr__(self, name):
		return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
	def __init__(self, env, num_repeats):
		self._env = env
		self._num_repeats = num_repeats

	def step(self, action):
		reward = 0.0
		discount = 1.0
		for i in range(self._num_repeats):
			self.t += 1
			time_step = self._env.step(action)
			reward += (time_step.reward or 0.0) * discount
			discount *= time_step.discount
			if time_step.last():
				break

		return time_step._replace(reward=reward, discount=discount)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def reset(self, seed=None, options=None):
		self.t = 0
		# print(type(self._env))
		return self._env.reset(seed=seed, options=options)

	def __getattr__(self, name):
		return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
	def __init__(self, env, num_frames, pixels_key='pixels'):
		self._env = env
		self._num_frames = num_frames
		self._frames = deque([], maxlen=num_frames)
		self._pixels_key = pixels_key

		wrapped_obs_spec = env.observation_spec()
		assert pixels_key in wrapped_obs_spec

		pixels_shape = wrapped_obs_spec[pixels_key].shape
		if len(pixels_shape) == 4:
			pixels_shape = pixels_shape[1:]
		self._obs_spec = specs.BoundedArray(shape=np.concatenate(
			[[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
											dtype=np.uint8,
											minimum=0,
											maximum=255,
											name='observation')

	def _transform_observation(self, time_step):
		assert len(self._frames) == self._num_frames
		obs = np.concatenate(list(self._frames), axis=0)
		return time_step._replace(observation=obs)

	def _extract_pixels(self, time_step):
		pixels = time_step.observation[self._pixels_key]
		if len(pixels.shape) == 4:
			pixels = pixels[0]
		return pixels.transpose(2, 0, 1).copy()

	def reset(self, seed=None, options=None):
		time_step = self._env.reset(seed=seed, options=options)
		pixels = self._extract_pixels(time_step)
		for _ in range(self._num_frames):
			self._frames.append(pixels)
		return self._transform_observation(time_step)

	def step(self, action):
		time_step = self._env.step(action)
		pixels = self._extract_pixels(time_step)
		self._frames.append(pixels)
		return self._transform_observation(time_step)

	def observation_spec(self):
		return self._obs_spec

	def action_spec(self):
		return self._env.action_spec()

	def __getattr__(self, name):
		return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
	def __init__(self, env, dtype, min_dt = None, max_dt = None):
		self._env = env
		wrapped_action_spec = env.action_spec()

		print(f"Original action spec: {wrapped_action_spec}")
		# Add time component to action spec
		if min_dt is None or max_dt is None:
			self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
											   dtype,
											   wrapped_action_spec.minimum,
											   wrapped_action_spec.maximum,
											   'action')
		else:
			self._action_spec = specs.BoundedArray((wrapped_action_spec.shape[0] + 1,),
											   dtype,
											   np.concatenate([wrapped_action_spec.minimum, [min_dt]]),
											   np.concatenate([wrapped_action_spec.maximum, [max_dt]]),
											   'action')

			print(f"Using TEA. Modified action spec: {self._action_spec}")

	def step(self, action):
		action = action.astype(self._env.action_spec().dtype)
		return self._env.step(action)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._action_spec

	def reset(self, seed=None, options=None):
		# print(type(self._env))
		return self._env.reset()

	def __getattr__(self, name):
		return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
	def __init__(self, env):
		self._env = env

	def reset(self, seed=None, options=None):
		# print(type(self._env))
		time_step = self._env.reset(seed=seed, options=options)
		return self._augment_time_step(time_step)

	def step(self, action):
		time_step = self._env.step(action)
		return self._augment_time_step(time_step, action)

	def _augment_time_step(self, time_step, action=None):
		if action is None:
			action_spec = self.action_spec()
			action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
		return ExtendedTimeStep(observation=time_step.observation,
								step_type=time_step.step_type,
								action=action,
								reward=time_step.reward or 0.0,
								discount=time_step.discount or 1.0)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def __getattr__(self, name):
		return getattr(self._env, name)


class TimeStepToGymWrapper(object):
	def __init__(self, env, domain, task, modality):
		try: # pixels
			obs_shp = env.observation_spec().shape
			assert modality == 'pixels'
		except: # state
			obs_shp = []
			for v in env.observation_spec().values():
				try:
					shp = np.prod(v.shape)
				except:
					shp = 1
				obs_shp.append(shp)
			obs_shp = (np.sum(obs_shp, dtype=np.int32),)
			assert modality != 'pixels'
		act_shp = env.action_spec().shape
		obs_dtype = np.float32 if modality != 'pixels' else np.uint8
		self.observation_space = gym.spaces.Box(
			low=np.full(
				obs_shp,
				-np.inf if modality != 'pixels' else env.observation_spec().minimum,
				dtype=obs_dtype),
			high=np.full(
				obs_shp,
				np.inf if modality != 'pixels' else env.observation_spec().maximum,
				dtype=obs_dtype),
			shape=obs_shp,
			dtype=obs_dtype,
		)
		self.action_space = gym.spaces.Box(
			low=np.full(act_shp, env.action_spec().minimum),
			high=np.full(act_shp, env.action_spec().maximum),
			shape=act_shp,
			dtype=env.action_spec().dtype)
		self.env = env
		self.domain = domain
		self.task = task
		self.ep_len = 1000
		self.modality = modality
		self.t = 0
	
	@property
	def unwrapped(self):
		return self.env

	@property
	def reward_range(self):
		return None

	@property
	def metadata(self):
		return None
	
	def _obs_to_array(self, obs):
		if self.modality != 'pixels':
			return np.concatenate([v.flatten() for v in obs.values()])
		return obs

	def reset(self, seed = None, options = None):
		self.t = 0
		# return self._obs_to_array(self.env.reset().observation)
		# print(type(self.env))
		return self._obs_to_array(self.env.reset(seed=seed, options = options).observation)
	
	def step(self, action):
		self.t += 1
		time_step = self.env.step(action)
		return self._obs_to_array(time_step.observation), time_step.reward, time_step.last() or self.t == self.ep_len, defaultdict(float)

	def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
		camera_id = dict(quadruped=2).get(self.domain, camera_id)
		return self.env.physics.render(height, width, camera_id)


class DefaultDictWrapper(gym.Wrapper):
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)

	def step(self, action):
		# print(f"[DefaultDictWrapper] Action : {action}")
		obs, reward, done, info = self.env.step(action)
		# print(obs.shape)
		return obs, reward, False, done, defaultdict(float, info)


def make_env(cfg):
	"""
	Make DMControl environment for TD-MPC experiments.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	domain, task = cfg.task.replace('-', '_').split('_', 1)
	domain = dict(cup='ball_in_cup').get(domain, domain)
	assert (domain, task) in suite.ALL_TASKS
	env = suite.load(domain,
					 task,
					 task_kwargs={'random': cfg.seed},
					 visualize_reward=False)

	if cfg.tea:
		env = ActionDTypeWrapper(env, np.float32, cfg.min_dt, cfg.max_dt)
		env = TemporallyExtendedActionWrapper(env, dt=env.physics.timestep())
	else:
		env = ActionDTypeWrapper(env, np.float32)
		env = ActionRepeatWrapper(env, cfg.action_repeat)
	env = ActionScaleWrapper(env, minimum=-1.0, maximum=+1.0)

	if cfg.modality=='pixels':
		if (domain, task) in suite.ALL_TASKS:
			camera_id = dict(quadruped=2).get(domain, 0)
			render_kwargs = dict(height=84, width=84, camera_id=camera_id)
			env = pixels.Wrapper(env,
								pixels_only=True,
								render_kwargs=render_kwargs)
		env = FrameStackWrapper(env, cfg.get('frame_stack', 1), cfg.modality)
	env = ExtendedTimeStepWrapper(env)
	env = TimeStepToGymWrapper(env, domain, task, cfg.modality)
	env = DefaultDictWrapper(env)

	# Convenience
	cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
	cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
	cfg.action_dim = env.action_space.shape[0]	
	print(f"Action space: {env.action_space}")
	return env



# Copyright 2019 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Wrapper that scales actions to a specific range."""

import dm_env
from dm_env import specs
import numpy as np

_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY = (
    "`env.action_spec()` must return a single `BoundedArray`, got: {}.")
_MUST_BE_FINITE = "All values in `{name}` must be finite, got: {bounds}."
_MUST_BROADCAST = (
    "`{name}` must be broadcastable to shape {shape}, got: {bounds}.")


class ActionScaleWrapper(dm_env.Environment):
  """Wraps a control environment to rescale actions to a specific range."""
  __slots__ = ("_action_spec", "_env", "_transform")

  def __init__(self, env, minimum, maximum):
    """Initializes a new action scale Wrapper.

    Args:
      env: Instance of `dm_env.Environment` to wrap. Its `action_spec` must
        consist of a single `BoundedArray` with all-finite bounds.
      minimum: Scalar or array-like specifying element-wise lower bounds
        (inclusive) for the `action_spec` of the wrapped environment. Must be
        finite and broadcastable to the shape of the `action_spec`.
      maximum: Scalar or array-like specifying element-wise upper bounds
        (inclusive) for the `action_spec` of the wrapped environment. Must be
        finite and broadcastable to the shape of the `action_spec`.

    Raises:
      ValueError: If `env.action_spec()` is not a single `BoundedArray`.
      ValueError: If `env.action_spec()` has non-finite bounds.
      ValueError: If `minimum` or `maximum` contain non-finite values.
      ValueError: If `minimum` or `maximum` are not broadcastable to
        `env.action_spec().shape`.
    """
    action_spec = env.action_spec()
    if not isinstance(action_spec, specs.BoundedArray):
      raise ValueError(_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY.format(action_spec))

    minimum = np.array(minimum)
    maximum = np.array(maximum)
    shape = action_spec.shape
    orig_minimum = action_spec.minimum
    orig_maximum = action_spec.maximum
    orig_dtype = action_spec.dtype

    print(f"ActionScaleWrapper: original action spec min {orig_minimum}, max {orig_maximum}")
    print(f"ActionScaleWrapper: new action spec min {minimum}, max {maximum}")

    def validate(bounds, name):
      if not np.all(np.isfinite(bounds)):
        raise ValueError(_MUST_BE_FINITE.format(name=name, bounds=bounds))
      try:
        np.broadcast_to(bounds, shape)
      except ValueError:
        raise ValueError(_MUST_BROADCAST.format(
            name=name, bounds=bounds, shape=shape))

    validate(minimum, "minimum")
    validate(maximum, "maximum")
    validate(orig_minimum, "env.action_spec().minimum")
    validate(orig_maximum, "env.action_spec().maximum")

    scale = (orig_maximum - orig_minimum) / (maximum - minimum)

    def transform(action):
      new_action = orig_minimum + scale * (action - minimum)
      return new_action.astype(orig_dtype, copy=False)

    dtype = np.result_type(minimum, maximum, orig_dtype)
    self._action_spec = action_spec.replace(
        minimum=minimum, maximum=maximum, dtype=dtype)
    self._env = env
    self._transform = transform

  def step(self, action):
    # print(f"[ActionScaleWrapper] Action : {action}")
    # print(f"[ActionScaleWrapper] After transform - Action : {self._transform(action)}")
    return self._env.step(self._transform(action))

  def reset(self, seed=None, options=None):
    return self._env.reset(seed=seed, options=options)

  def observation_spec(self):
    return self._env.observation_spec()

  def action_spec(self):
    return self._action_spec

  def __getattr__(self, name):
    return getattr(self._env, name)
