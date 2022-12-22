import json
import os
import argparse
import torch
import numpy as np
from mpi4py import MPI
from risk.common.mpi_data_utils import mpi_gather_objects
from risk.cdf_rl.cdf_policy_optimizer import CDFPolicyOptimizer
from risk.rl.constrained_policy_optimizer import ConstrainedPolicyOptimizer


# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


class ConstrainedCDFPolicyOptimizer(ConstrainedPolicyOptimizer, CDFPolicyOptimizer):

    def process_config(self):
        """  Processes configuration, filling in missing values as appropriate.  """
        if self.config['trpo']:
            self.config.setdefault('surrogate', False)             # Use TRPO-style update, but with risk-sensitive loss
        super().process_config()
        if 'mean' not in self.config['metrics']:
            self.config['metrics'].append({'type': 'mean'})        # Include mean of reward with changing penalty
        self.config['utility'].setdefault('c_reference', 0)        # Cost utility reference point (only used in CPT)

    def process_episode(self, episode_buffer, episode_info):
        """  Processes a completed episode, storing required data in buffer  """
        if self.config['utility']['variable_ref']:  # compute later, once we have rewards for whole batch
            r_utilities, r_q_values, c_utilities, c_q_values = None, None, None, None
        else:
            r_utilities = self.compute_utilities(episode_buffer[:, 2], episode_info)
            c_utilities = self.compute_utilities(episode_buffer[:, 7], episode_info)
            r_q_values = self.compute_target_values(r_utilities) if self.mode == 'train' else None
            c_q_values = self.compute_target_values(c_utilities) if self.mode == 'train' else None
        self.buffer.update(episode_buffer, q_values=r_q_values, utilities=r_utilities,
                           c_q_values=c_q_values, c_utilities=c_utilities)

    def estimate_advantages(self):
        super().estimate_advantages()
        # Update weights, utilities:
        self.compute_weights()
        self.update_utilities()

    def compute_weights(self):
        """  Collect episode rewards on one worker, compute weights, update worker buffers  """
        # Collect rewards and episode breakpoints:
        all_rewards = mpi_gather_objects(MPI.COMM_WORLD, self.buffer.rewards)
        all_dones = mpi_gather_objects(MPI.COMM_WORLD, self.buffer.dones)
        all_costs = mpi_gather_objects(MPI.COMM_WORLD, self.buffer.costs)
        if self.id == 0:  # only need to compute weights on one worker
            # Sum episode rewards, including scaled cost:
            ep_total_rewards = []
            all_indices = []
            current_ind = 0
            for i in range(len(all_rewards)):
                terminal_ind = np.where(all_dones[i])[0]
                episode_ind = np.ones(all_rewards[i].shape)
                for j in range(terminal_ind.shape[0]):
                    start_ind = 0 if j == 0 else terminal_ind[j-1] + 1
                    stop_ind = terminal_ind[j] + 1
                    ep_total = np.sum(all_rewards[i][start_ind:stop_ind]) \
                        - self.penalty*np.sum(all_costs[i][start_ind:stop_ind])
                    ep_total_rewards.append(ep_total)
                    episode_ind[start_ind:stop_ind] *= current_ind
                    current_ind += 1
                all_indices.append(episode_ind.astype(int))
            # Compute weights:
            episode_weights = self.weight_coefficient_function(ep_total_rewards)
            all_weights = [episode_weights[all_indices[i]] for i in range(len(all_indices))]
            # Update utility reference:
            self.update_utility_reference(ep_total_rewards)
        else:
            ep_total_rewards, all_weights = [], []
        # Distribute back to workers:
        all_weights = MPI.COMM_WORLD.bcast(all_weights, root=0)
        self.buffer.weights = all_weights[self.id]
        self.config['batch_size'] = self.config['true_batch_size']
        self.config = MPI.COMM_WORLD.bcast(self.config, root=0)  # sync utility references, batch size

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
                r_utilities = self.utility_function(episode_rewards, self.config['utility']['reference'])
                q_values = self.compute_target_values(r_utilities)
                self.buffer.r_utilities = np.concatenate((self.buffer.r_utilities, r_utilities))
                self.buffer.q_values = np.concatenate((self.buffer.q_values, q_values))
                episode_costs = -self.penalty_scale * self.buffer.costs[start_ind:stop_ind]
                c_utilities = self.utility_function(episode_costs, self.config['utility']['c_reference'])
                c_q_values = self.compute_target_values(c_utilities)
                self.buffer.c_utilities = np.concatenate((self.buffer.c_utilities, c_utilities))
                self.buffer.c_q_values = np.concatenate((self.buffer.c_q_values, c_q_values))

    def compute_metrics(self, episode_data):
        """  Computes metrics to be evaluated as learning progresses  """
        # Collect rewards:
        rewards = mpi_gather_objects(MPI.COMM_WORLD, episode_data['episode_reward'])
        rewards = self.flatten_list(rewards)
        costs = mpi_gather_objects(MPI.COMM_WORLD, episode_data['episode_cost'])
        costs = self.flatten_list(costs)
        total = [rewards[i] - self.penalty_scale * costs[i] for i in range(len(costs))]
        # Compute metrics on worker 0:
        metrics = {}
        if self.id == 0 and len(self.config['metrics']) > 0:
            if self.mode.lower() == 'train' and self.config['rewards_stored'] > 0:
                for i, metric in enumerate(self.config['metrics']):
                    metrics[metric['type']] = self.metric_functions[i](total)
            else:
                for i, metric in enumerate(self.config['metrics']):
                    metrics[metric['type']] = self.metric_functions[i](total)
        metrics = MPI.COMM_WORLD.bcast(metrics, root=0)
        return metrics

    def test(self):
        test_output = super().test()
        evaluation_metrics = self.compute_metrics(test_output)
        updated_output = {**evaluation_metrics, **test_output}
        self.store_test_results(updated_output)


if __name__ == '__main__':
    """  Runs ConstrainedCDFPolicyOptimizer training or testing for a given input configuration file  """
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
        c_cdf_po_object = ConstrainedCDFPolicyOptimizer(config1)
        c_cdf_po_object.train()
    else:
        config1['use_prior_nets'] = True
        c_cdf_po_object = ConstrainedCDFPolicyOptimizer(config1)
        c_cdf_po_object.test()
