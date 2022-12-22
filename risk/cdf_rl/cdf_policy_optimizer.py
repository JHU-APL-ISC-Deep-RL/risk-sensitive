import json
import os
import argparse
import torch
import numpy as np
from mpi4py import MPI
from copy import deepcopy
from risk.common.mpi_data_utils import mpi_gather_objects, mpi_sum, mpi_statistics_scalar
from risk.rl.policy_optimizer import PolicyOptimizer
from risk.cdf_rl.utility_functions import load_utility_function
from risk.cdf_rl.weight_coefficients import load_weight_coefficient_function
from risk.cdf_rl.metrics import load_metrics

# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


class CDFPolicyOptimizer(PolicyOptimizer):

    def __init__(self, config, weight_function=None):
        self.penalty_scale = 0
        super().__init__(config)
        self.utility_function = load_utility_function(self.config['utility'])
        self.weight_coefficient_function = load_weight_coefficient_function(self.config['weight'], weight_function)
        self.metric_functions = load_metrics(self.config['metrics'])
        self.total_cost = 0
        self.total_trajectories = 0
        self.reward_history = []

    def process_config(self):
        """  Processes configuration, filling in missing values as appropriate.  """
        self.config.setdefault('gamma', 1.0)                       # Default is to not discount
        if self.config['trpo']:
            self.config.setdefault('surrogate', False)             # Use TRPO-style update, but with risk-sensitive loss
        super().process_config()                                   # General policy gradient configuration
        if 'mod_config' in self.config['environment']:
            self.penalty_scale = self.config['environment']['mod_config']['scale']
        self.config.setdefault('utility', {'type': 'identity'})    # Utility configuration
        self.config['utility'].setdefault('reference', 0)          # Utility reference point (only used in CPT)
        self.config['utility'].setdefault('variable_ref', False)   # May not be legit, but for exploring variable ref.
        self.config['utility'].setdefault('include_info', False)   # Whether environment info needed for utility
        self.config.setdefault('weight', {'type': 'uniform'})      # Weight configuration
        self.config['weight'].setdefault('reference', 0)           # Utility reference point (only used in CPT)
        self.config['weight'].setdefault('variable_ref', False)    # Whether utility has a variable reference point
        self.config.setdefault('metrics', [])                      # Metrics to compute, plot on evaluation
        self.config.setdefault('rewards_stored', 0)                # Number of episode rewards stored
        # Configure warmup if rewards are stored:
        self.config['true_batch_size'] = self.config['batch_size']
        if self.config['rewards_stored'] > 0:
            self.config['batch_size'] += self.config['rewards_stored'] * self.config['max_ep_length']
        # Check consistency:
        assert self.config['utility']['reference'] == self.config['weight']['reference'], 'reference config'
        assert self.config['utility']['variable_ref'] == self.config['weight']['variable_ref'], 'variable_ref config'

    def process_episode(self, episode_buffer, episode_info):
        """  Processes a completed episode, storing required data in buffer  """
        if self.config['utility']['variable_ref']:  # compute later, once we have rewards for whole batch
            utilities, q_values = None, None
        else:
            utilities = self.compute_utilities(episode_buffer[:, 2], episode_info)
            q_values = self.compute_target_values(utilities) if self.mode == 'train' else None
        self.buffer.update(episode_buffer, q_values, utilities=utilities)

    def compute_utilities(self, rewards, utility_info: dict):
        """  Transform episode rewards to utilities, returning updated episode buffer  """
        if self.config['utility']['include_info']:
            return self.utility_function(rewards, **utility_info)
        else:
            return self.utility_function(rewards)

    def update_network(self):
        """  Updates the networks based on processing from all workers  """
        self.compute_weights()
        self.update_utilities()  # only needed if reference is variable
        return super().update_network()

    def compute_weights(self):
        """  Collect episode rewards on one worker, compute weights, update worker buffers  """
        # Collect rewards and episode breakpoints:
        all_rewards = mpi_gather_objects(MPI.COMM_WORLD, self.buffer.rewards)
        all_dones = mpi_gather_objects(MPI.COMM_WORLD, self.buffer.dones)
        if self.id == 0:  # only need to compute weights on one worker
            # Sum episode rewards:
            episode_rewards = []
            all_indices = []
            current_ind = 0
            for i in range(len(all_rewards)):
                terminal_ind = np.where(all_dones[i])[0]
                episode_ind = np.ones(all_rewards[i].shape)
                for j in range(terminal_ind.shape[0]):
                    start_ind = 0 if j == 0 else terminal_ind[j-1] + 1
                    stop_ind = terminal_ind[j] + 1
                    episode_rewards.append(np.sum(all_rewards[i][start_ind:stop_ind]))
                    episode_ind[start_ind:stop_ind] *= current_ind
                    current_ind += 1
                all_indices.append(episode_ind.astype(int))
            # Compute weights:
            episode_weights = self.weight_coefficient_function(episode_rewards + self.reward_history)
            all_weights = [episode_weights[all_indices[i]] for i in range(len(all_indices))]
            # Update utility reference:
            self.update_utility_reference(episode_rewards + self.reward_history)
        else:
            episode_rewards, all_weights = [], []
        # Distribute back to workers:
        all_weights = MPI.COMM_WORLD.bcast(all_weights, root=0)
        self.buffer.weights = all_weights[self.id]
        self.config['batch_size'] = self.config['true_batch_size']
        self.config = MPI.COMM_WORLD.bcast(self.config, root=0)  # sync utility references, batch size

    def update_reward_history(self, episode_rewards):
        """  Updates database of recently-obtained episode rewards (for computing weights; only used on worker 0)  """
        if self.config['rewards_stored'] > 0:
            self.reward_history += episode_rewards
            self.reward_history = self.reward_history[-self.config['rewards_stored']:]

    def update_utility_reference(self, episode_rewards):
        """  Updates utility reference point in the event that it is variable  """
        if self.config['utility']['variable_ref']:
            self.config['utility']['reference'] = np.mean(episode_rewards)

    def update_utilities(self):
        """  Update utility estimates using reference point based on current batch of rewards  """
        if self.config['utility']['variable_ref']:
            terminal_ind = np.where(self.buffer.dones)[0]
            for i in range(terminal_ind.shape[0]):
                start_ind = 0 if i == 0 else terminal_ind[i - 1] + 1
                stop_ind = terminal_ind[i] + 1
                episode_rewards = self.buffer.rewards[start_ind:stop_ind]
                utilities = self.utility_function(episode_rewards, self.config['utility']['reference'])
                q_values = self.compute_target_values(utilities)
                self.buffer.utilities = np.concatenate((self.buffer.utilities, utilities))
                self.buffer.q_values = np.concatenate((self.buffer.q_values, q_values))

    def estimate_advantages(self):
        """  Estimate advantages for a sequence of observations and rewards  """
        if not self.config['reward_to_go']:
            self.buffer.advantages = self.buffer.q_values
        else:
            if 'v_network' in self.config:
                if self.config['gae']:
                    utilities, values, dones = self.buffer.utilities, self.buffer.values, self.buffer.dones
                    self.buffer.advantages = self.estimate_generalized_advantage(utilities, values, dones)
                else:
                    self.buffer.advantages = self.buffer.q_values - self.buffer.values
            else:
                self.buffer.advantages = deepcopy(self.buffer.q_values)
        mean_adv, std_adv = mpi_statistics_scalar(MPI.COMM_WORLD, self.buffer.advantages)
        self.buffer.advantages = (self.buffer.advantages - mean_adv) / std_adv
        return mean_adv, std_adv

    def compute_policy_loss(self, observations, actions, advantages, old_log_probs, clip=True):
        """  Compute policy loss, entropy, kld  """
        pi_loss, entropy, kld = super().compute_policy_loss(observations, actions, advantages, old_log_probs, clip)
        weights = torch.from_numpy(self.buffer.weights.astype(float)).float()
        pi_loss = pi_loss * weights
        return pi_loss, entropy, kld

    def compute_metrics(self, episode_data):
        """  Computes metrics to be evaluated as learning progresses  """
        # Collect rewards:
        rewards = mpi_gather_objects(MPI.COMM_WORLD, episode_data['episode_reward'])
        rewards = self.flatten_list(rewards)
        # Compute metrics on worker 0:
        metrics = {}
        if self.id == 0 and len(self.config['metrics']) > 0:
            if self.mode.lower() == 'train' and self.config['rewards_stored'] > 0:
                for i, metric in enumerate(self.config['metrics']):
                    metrics[metric['type']] = self.metric_functions[i](rewards + self.reward_history)
            else:
                for i, metric in enumerate(self.config['metrics']):
                    metrics[metric['type']] = self.metric_functions[i](rewards)
        metrics = MPI.COMM_WORLD.bcast(metrics, root=0)
        # Update reward history:
        if self.id == 0:
            self.update_reward_history(rewards)
        return metrics

    def update_logging(self, episode_summaries, losses, evaluation, steps, previous_steps):
        """  Updates TensorBoard logging based on most recent update  """
        current_steps = self.log_rewards_costs(episode_summaries, steps, previous_steps)
        training_metrics = self.compute_metrics(episode_summaries)
        if self.id == 0:
            for k, v in training_metrics.items():
                self.logger.summary_writer.add_scalar('Performance/' + k, v, current_steps + previous_steps)
        super().update_logging(episode_summaries, losses, evaluation, steps, previous_steps)

    def log_rewards_costs(self, episode_summaries, steps, previous_steps,):
        """  Log mean cost over training to date, as well as positive contribution to reward  """
        positive_rewards = deepcopy(episode_summaries['episode_reward'])
        for k, v in episode_summaries.items():
            if k == 'cost':
                self.total_cost += sum(v)
                positive_rewards = [positive_rewards[i] + v[i]*self.penalty_scale for i in range(len(v))]
        self.total_trajectories += self.buffer.trajectories
        overall_total_cost = mpi_sum(MPI.COMM_WORLD, self.total_cost)
        overall_total_trajectories = mpi_sum(MPI.COMM_WORLD, self.total_trajectories)
        cost_rate = overall_total_cost / overall_total_trajectories
        current_steps = mpi_sum(MPI.COMM_WORLD, steps)
        if self.id == 0:
            self.logger.summary_writer.add_scalar('Performance/cost_rate', cost_rate, current_steps + previous_steps)
        self.logger.log_mean_value('Performance/positive_rewards', positive_rewards, steps, previous_steps)
        return current_steps

    def test(self):
        test_output = super().test()
        evaluation_metrics = self.compute_metrics(test_output)
        updated_output = {**evaluation_metrics, **test_output}
        self.store_test_results(updated_output)

    @staticmethod
    def flatten_list(nested_list):
        return [item for sublist in nested_list for item in sublist]


if __name__ == '__main__':
    """  Runs CDFPolicyOptimizer training or testing for a given input configuration file  """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file to run', required=True)
    parser.add_argument('--mode', default='train', required=False, help='mode ("train" or "test")')
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=0)
    parser.add_argument('--trpo', help='whether to force trpo update', required=False, type=int, default=0)
    in_args = parser.parse_args()
    with open(os.path.join(os.getcwd(), in_args.config), 'r') as f1:
        config1 = json.load(f1)
    config1['seed'] = in_args.seed
    if 'trpo' not in config1:
        config1['trpo'] = bool(in_args.trpo)
    if in_args.mode.lower() == 'train':
        cdf_po_object = CDFPolicyOptimizer(config1)
        cdf_po_object.train()
    else:
        config1['use_prior_nets'] = True
        cdf_po_object = CDFPolicyOptimizer(config1)
        cdf_po_object.test()
