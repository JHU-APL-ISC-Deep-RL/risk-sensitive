import os
import configparser
import torch
import numpy as np
from copy import deepcopy
from typing import Dict


def get_env_object(config: Dict):
    """
    Helper function that returns environment object.  Can include more environments as they become available.  While
    the following does not explicitly require that the environment inherit from gym.Env, any environment that does
    follow the OpenAI gym format should be compatible.
    """
    if 'environment' not in config:
        raise ValueError('environment information missing from config')
    if config['environment']['name'].lower() == 'crowdsim':
        from crowd_sim.envs.utils.robot import Robot
        from crowd_sim.envs import CrowdSim
        from crowd_sim.envs.utils.state import JointState

        class CrowdSimReformatted(CrowdSim):

            def reset(self, phase='train', test_case=None):
                observation = super().reset(phase, test_case)
                if self.robot.sensor == 'coordinates':
                    observation = JointState(self.robot.get_full_state(), observation)
                return observation

            def step(self, action, update=True):
                action = self.robot.policy.action_space[action]
                observation, reward, done, info = super().step(action, update)
                if self.robot.sensor == 'coordinates':
                    observation = JointState(self.robot.get_full_state(), observation)
                return observation, reward, done, info

        env_config = configparser.RawConfigParser()
        env_config_file = os.path.join(os.getcwd(), '../../envs/CrowdNav/crowd_nav/configs/',
                                       config['environment']['env_config_file'])
        env_config.read(env_config_file)
        env = CrowdSimReformatted()  # env = gym.make('CrowdSim-v0') for original
        env.configure(env_config)
        robot = Robot(env_config, 'robot')
        env.set_robot(robot)
        return env
    else:
        import gym
        if config['environment']['type'].lower() == 'atari':
            from .atari_wrappers import make_atari, wrap_deepmind

            class PyTorchAtari(gym.Wrapper):
                def __init__(self, base_env, dim_order: tuple):
                    """  Wrapper to appropriately re-shape arrays for PyTorch processing  """
                    gym.Wrapper.__init__(self, base_env)
                    self.dim_order = dim_order

                def reset(self, **kwargs):
                    obs = self.env.reset(**kwargs)
                    return np.transpose(obs, (0, 3, 1, 2))

                def step(self, action):
                    obs, reward, done, info = self.env.step(action)
                    return np.transpose(obs, (0, 3, 1, 2)), reward, done, info

            env_config = deepcopy(config['environment'])
            env_config['clip_rewards'] = 0
            env_name = env_config.pop('name', None)
            return PyTorchAtari(wrap_deepmind(make_atari(env_name), **env_config), (0, 3, 1, 2))
        else:
            if config['environment']['type'].lower() == 'gym':  # Load OpenAI gym environment

                class ScaledGym(gym.Wrapper):
                    def __init__(self, base_env, reward_scale):
                        """
                        Scales returned rewards from gym environment
                        :param base_env: (Gym Environment); the environment to wrap
                        :param reward_scale: (float); multiplier for reward
                        """
                        gym.Wrapper.__init__(self, base_env)
                        self.reward_scale = reward_scale

                    def step(self, action):
                        obs, reward, done, info = self.env.step(action)
                        return obs, reward * self.reward_scale, done, info

                if 'scale' in config['environment']:
                    env = ScaledGym(gym.make(config['environment']['name']), config['environment']['scale'])
                else:
                    env = gym.make(config['environment']['name'])
            else:
                import safety_gym

                class ModifiedSafetyGym(gym.Wrapper):
                    def __init__(self, base_env, mod_dict):
                        """
                        Makes minor modifications to default SafetyGym configuration
                        :param base_env: (Gym Environment); the environment to wrap
                        :param mod_dict: (dict); configuration for modified SafetyGym
                        """
                        gym.Wrapper.__init__(self, base_env)
                        self.mod_config = mod_dict  # may expand on this later
                        # Optionally enable a discrete action space:
                        self.bins = np.array([])
                        self.total_bins = 0
                        self.mod_config.setdefault('discrete', 0)
                        if self.mod_config['discrete'] > 1:
                            if 'doggo' in self.robot_base:
                                raise NotImplementedError('Cannot use discrete action space with doggo robot.')
                            self.bins = np.arange(-1, 1 + 1e-5, 2 / (self.mod_config['discrete']-1))
                            self.total_bins = len(self.bins)
                        # Configure cost/penalty incorporation in reward:
                        self.mod_config.setdefault('cost', 'one_indicator')  # one_indicator in default SafetyGym
                        assert self.mod_config['cost'] in ['full', 'all_indicators', 'one_indicator']
                        if self.mod_config['cost'] == 'full':
                            self.unwrapped.config['constrain_indicator'] = False
                            self.unwrapped.constrain_indicator = False  # True by default
                        self.mod_config.setdefault('scale', 0)

                    def step(self, action: int):
                        if self.mod_config['discrete'] > 1:
                            action = np.array([self.bins[action // self.total_bins],
                                               self.bins[action % self.total_bins]])
                        obs, reward, done, info = self.env.step(action)
                        if self.mod_config['cost'] == 'all_indicators':
                            info['cost'] = 0
                            for k, v in info.items():
                                if k[:4] == 'cost' and k != 'cost':  # just count once
                                    reward -= v * self.mod_config['scale']
                                    info['cost'] += v
                        else:
                            cost = info.get('cost', 0)
                            reward -= cost * self.mod_config['scale']
                        return obs, reward, done, info

                if config['environment']['type'].lower() == 'safety_engine':  # Custom Safety Gym environment
                    from safety_gym.envs.engine import Engine
                    config_dict = deepcopy(config['environment'])
                    del config_dict['type']
                    if 'mod_config' in config_dict:
                        mod_config = deepcopy(config_dict['mod_config'])
                        del config_dict['mod_config']
                        env = ModifiedSafetyGym(Engine(config=config_dict), mod_config)
                    else:
                        env = Engine(config=config_dict)
                else:
                    assert config['environment']['type'].lower()[:4] == 'safe', "Unknown environment type."
                    env = gym.make(config['environment']['name'])
                    mod_config = deepcopy(config['environment']['mod_config'])
                    env = ModifiedSafetyGym(env, mod_config)
            config['pi_network']['obs_dim'] = env.observation_space.shape[0]
            if config['pi_network']['discrete']:
                config['pi_network']['action_dim'] = env.total_bins ** 2  # currently, only works for ConfigSafetyGym
            else:
                config['pi_network']['action_dim'] = env.action_space.shape[0]
            if 'v_network' in config:
                config['v_network']['obs_dim'] = env.observation_space.shape[0]
                config['v_network']['action_dim'] = 1
            return env


def get_network_object(config: Dict, env=None) -> torch.nn.Module:
    """  Helper function that returns network object.  Can include more networks as they become available.  """
    if 'network_name' not in config:
        raise ValueError('network_name missing from config')
    if config['network_name'].lower() == 'atari':
        from .networks import AtariNetwork
        return AtariNetwork(config)
    elif config['network_name'].lower() == 'crowdnavpixel':
        from .networks import CrowdNavConv
        config.setdefault('humans_use_net', False)
        if env is not None:
            config['v_pref'] = env.robot.v_pref
            assert config['action_dim'] == int(env.config.get('action_space', 'speed_samples')) * \
                int(env.config.get('action_space', 'rotation_samples')) + 1
            policy = CrowdNavConv(config, env.config)
            env.robot.set_policy(policy)
            if config['humans_use_net']:
                for human in env.humans:
                    human.set_policy(policy)  # can make this configurable
        else:
            policy = CrowdNavConv(config)
        return policy
    elif config['network_name'].lower() == 'sarl_pv':
        from crowd_nav.policy.sarl_pv import SARLPV
        config.setdefault('humans_use_net', False)
        policy_config = configparser.RawConfigParser()
        policy_config.read(os.path.join(os.getcwd(), '../envs/CrowdNav/crowd_nav/configs/',
                                        config['network_config_file']))
        policy = SARLPV(policy_config)
        if env is not None:
            policy.set_env(env)
            env.robot.set_policy(policy)
            if config['humans_use_net']:
                for human in env.humans:
                    human.set_policy(policy)  # can make this configurable
        return policy
    elif config['network_name'].lower() == 'mlp':
        from .networks import MLP
        return MLP(config)
    elif config['network_name'].lower() == 'mlp_categorical':
        from .networks import CategoricalMLP
        return CategoricalMLP(config)
    elif config['network_name'].lower() == 'mlp_gaussian':
        from .networks import GaussianMLP
        return GaussianMLP(config)
    else:
        raise ValueError('network_name not recognized.')


def get_sampler(config: Dict, deterministic=False):
    suffix = config['network_name'].lower().split('_')[-1]
    if suffix == 'categorical':
        from .samplers import CategoricalSampler
        return CategoricalSampler(config, deterministic)
    elif suffix == 'gaussian':
        from .samplers import GaussianSampler
        return GaussianSampler(config, deterministic)
    elif suffix == 'beta':
        from .samplers import BetaSampler
        return BetaSampler(config, deterministic)
