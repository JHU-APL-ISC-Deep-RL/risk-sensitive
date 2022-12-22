import torch
import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.special import gamma, digamma


def compute_entropy(policy, lb=-1, ub=1, inc=0.01):
    mean = policy[0].detach().numpy()
    std = policy[1].detach().numpy()

    def prob(x, mu, sigma):
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2) ** .5

    entropies = []
    for i in range(len(mean)):
        bins = np.arange(mean[i] - 4 * std[i], mean[i] + 4 * std[i], inc)
        probs = prob(bins, mean[i], std[i])
        mp = []
        for j in range(len(bins)):
            if lb < bins[j] < ub:
                mp.append(probs[j])
        mp = np.array(mp)
        entropies.append(-np.sum(mp * inc * np.log(mp)))
    return np.mean(entropies)


class BaseSampler(object):
    __metaclass__ = ABCMeta

    def __init__(self, config, deterministic):
        self.epsilon = 1.e-6
        self.config = config
        self.deterministic = deterministic

    def get_raw_action(self, action):
        """  Convert action seen by the environment to raw action sampled from network  """
        return action

    @abstractmethod
    def get_action_and_log_prob(self, pi):
        """  Sample action based on neural network output  """
        raise NotImplementedError('Sampler must have get_action_and_log_prob method.')

    @abstractmethod
    def compute_kld(self, old_pi, new_pi):
        """  Return average KL divergence between old and new policies  """
        raise NotImplementedError('Sampler must have compute_kld method.')

    @staticmethod
    def get_distribution(pi):
        """  Return Torch distribution object as described by policy  """
        pass

    @staticmethod
    def compute_entropy(pi):
        """  Return average entropy of distributions described by policy  """
        pass

    @staticmethod
    def np_entropy(pi):
        """  Returns entropies of each row of a numpy array of policies  """
        pass


class CategoricalSampler(BaseSampler):

    def get_action_and_log_prob(self, pi):
        if not self.deterministic:
            pi = pi.detach().numpy()
            if len(pi.shape) > 1:
                pi = np.squeeze(pi)
            action = int(np.random.choice(np.arange(pi.shape[-1]), p=pi))
            log_prob = np.log(pi[action])
        else:
            log_prob = None  # Not needed for testing
            action = int(np.argmax(pi.detach().numpy()))
        return action, log_prob

    def compute_kld(self, old_pi, new_pi):
        new_pi = new_pi.detach().numpy()
        all_terms = new_pi * (np.log(new_pi) - np.log(old_pi))
        return np.mean(np.sum(all_terms, axis=1))

    @staticmethod
    def compute_entropy(pi):
        return -torch.mean(pi * torch.log(pi))

    @staticmethod
    def np_entropy(pi):
        raise NotImplementedError('Not yet implemented!')


class GaussianSampler(BaseSampler):

    def __init__(self, config, deterministic):
        super().__init__(config, deterministic)
        self.config.setdefault('act_scale', 1)
        self.config.setdefault('act_offset', 0)

    def get_action_and_log_prob(self, pi):
        if not self.deterministic:
            pi_distribution = self.get_distribution(pi)
            sample = pi_distribution.sample()
            action = sample.detach().numpy()
            if self.config['bound_corr']:
                below = pi_distribution.cdf(sample.clamp(-1.0, 1.0))
                below = below.clamp(self.epsilon, 1.0).log().detach().numpy() * (action <= -1).astype(float)
                above = (torch.ones(sample.size()) - pi_distribution.cdf(sample.clamp(-1.0, 1.0)))
                above = above.clamp(self.epsilon, 1.0).log().detach().numpy() * (action >= 1).astype(float)
                in_bounds = pi_distribution.log_prob(sample).detach().numpy() * \
                    ((action > -1).astype(float)) * ((action < 1).astype(float))
                log_prob = np.sum(in_bounds + below + above)
            else:
                log_prob = np.sum(pi_distribution.log_prob(sample).detach().numpy())
        else:
            log_prob = None  # Not needed for testing
            action = pi[0].detach().numpy()
        return action * self.config['act_scale'] + self.config['act_offset'], log_prob

    def get_raw_action(self, action):
        return (action - self.config['act_offset']) / self.config['act_scale']

    def compute_kld(self, old_pi, new_pi):
        mu_old, sigma_old = np.split(old_pi, 2, axis=1)
        mu_new, sigma_new = new_pi[0].detach().numpy(), new_pi[1].detach().numpy()
        if not self.config['log_std_net']:
            sigma_new = np.repeat(np.expand_dims(sigma_new, 0), sigma_old.shape[0], axis=0)
        var_old, var_new = sigma_old ** 2, sigma_new ** 2
        all_kld = np.log(sigma_new / sigma_old) + 0.5 * (((mu_new - mu_old) ** 2 + var_old) / (var_new + 1.e-8) - 1)
        return np.mean(np.sum(all_kld, axis=1))

    @staticmethod
    def get_distribution(pi):
        return torch.distributions.Normal(pi[0], pi[1])

    @staticmethod
    def restore_distribution(pi):
        pi = np.split(pi, 2, axis=1)
        return torch.distributions.Normal(torch.from_numpy(pi[0]), torch.from_numpy(pi[1]))

    @staticmethod
    def compute_entropy(pi):
        return torch.mean(torch.log(pi[1]) + .5 * np.log(2 * np.pi * np.e))

    @staticmethod
    def np_entropy(pi):
        return np.mean(np.log(np.split(pi, 2, axis=1)[1]), axis=1) + .5 * np.log(2 * np.pi * np.e)


class BetaSampler(BaseSampler):

    def __init__(self, config, deterministic):
        super().__init__(config, deterministic)
        self.config['act_scale'] = 2
        self.config['act_offset'] = -1

    def get_action_and_log_prob(self, pi):
        if not self.deterministic:
            pi_distribution = self.get_distribution(pi)
            sample = pi_distribution.sample()
            action = sample.detach().numpy()
            log_prob = np.sum(pi_distribution.log_prob(sample).detach().numpy())
        else:
            log_prob = None  # Not needed for testing
            alpha = pi[0].detach().numpy()
            beta = pi[1].detach().numpy()
            action = (np.ones(alpha.shape()) - alpha) / (2*np.ones(alpha.shape()) - alpha - beta)
        return action * self.config['act_scale'] + self.config['act_offset'], log_prob

    def get_raw_action(self, action):
        return (action - self.config['act_offset']) / self.config['act_scale']

    def compute_kld(self, old_pi, new_pi):
        alpha_old, beta_old = np.split(old_pi, 2, axis=1)
        b_old = gamma(alpha_old) * gamma(beta_old) / gamma(alpha_old + beta_old)
        alpha_new, beta_new = new_pi[0].detach().numpy(), new_pi[1].detach().numpy()
        b_new = gamma(alpha_new) * gamma(beta_new) / gamma(alpha_new + beta_new)
        all_kld = np.log(b_new / b_old) + (alpha_old - alpha_new) * digamma(alpha_old) + \
            (beta_old - beta_new) * digamma(beta_old) + \
            (alpha_new - alpha_old + beta_new - beta_old) * gamma(alpha_old + beta_old)
        return np.mean(np.sum(all_kld, axis=1))

    @staticmethod
    def get_distribution(pi):
        return torch.distributions.Beta(pi[0], pi[1])

    @staticmethod
    def compute_entropy(pi):
        b = torch.exp(torch.lgamma(pi[0])) * torch.exp(torch.lgamma(pi[1])) / \
            torch.exp(torch.lgamma(pi[0] + pi[1]))
        return torch.mean(torch.log(b) - (pi[0] - torch.ones(pi[0].shape)) * torch.digamma(pi[0]) -
                          (pi[1] - torch.ones(pi[1].shape)) * torch.digamma(pi[1]) +
                          (pi[0] + pi[1] - 2 * torch.ones(pi[0].shape)) * torch.digamma(pi[0] + pi[1]))

    @staticmethod
    def np_entropy(pi):
        alpha, beta = np.split(pi, 2, axis=1)
        b = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
        all_entropies = np.log(b) - (alpha - 1) * digamma(alpha) - (beta - 1) * digamma(beta) \
            + (alpha + beta - 2) * digamma(alpha + beta)
        return np.mean(all_entropies, axis=1)
