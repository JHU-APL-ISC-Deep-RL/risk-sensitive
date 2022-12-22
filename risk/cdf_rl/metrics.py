import numpy as np
from scipy.stats import norm


def load_metrics(metric_list):
    """  Returns functions to compute relevant CDF-RL metrics  """

    metrics = []
    for config in metric_list:
        if config['type'].lower() == 'mean':

            def compute_mean(rewards):
                """  Returns mean reward  """
                return sum(rewards) / len(rewards)

            metrics.append(compute_mean)

        elif config['type'].lower() == 'cpt':
            config.setdefault('reference', 0)
            config.setdefault('variable_ref', False)
            config.setdefault('lambda', 2.25)
            config.setdefault('sigma', 0.88)
            config.setdefault('ep', .61)
            config.setdefault('em', .69)
            ref0, v_ref = config['reference'], config['variable_ref']
            lam, sig, ep, em = config['lambda'], config['sigma'], config['ep'], config['em']

            def w_plus(p):
                return p ** ep / ((p ** ep + (1 - p) ** ep) ** (1 / ep))

            def w_minus(p):
                return p ** em / ((p ** em + (1 - p) ** em) ** (1 / em))

            def compute_cpt(rewards):
                """  Estimates CPT value based on rewards  """
                ref = np.mean(rewards) if v_ref else ref0
                sorted_rewards = sorted(rewards)
                utilities = [((reward >= ref) - lam * (reward < ref)) * np.abs(reward - ref)**sig
                             for reward in sorted_rewards]
                n = len(rewards)
                c_est = 0
                for i in range(n):
                    c_plus = (utilities[i] >= 0) * utilities[i] * (w_plus((n - i) / n) - w_plus((n - i - 1) / n))
                    c_minus = (utilities[i] < 0) * utilities[i] * (w_minus((i + 1) / n) - w_minus(i / n))
                    c_est += (c_plus + c_minus)
                return c_est

            metrics.append(compute_cpt)

        elif config['type'].lower() == 'cvar':
            config.setdefault('alpha', 0.05)
            alpha = config['alpha']

            def compute_cvar(rewards):
                """  Estimates CVaR based on rewards  """
                num_trajectories = len(rewards)
                num_to_keep = int(num_trajectories * alpha)
                remainder = num_trajectories * alpha - num_to_keep
                num_to_keep += int(np.random.rand() <= remainder)
                sorted_rewards = sorted(rewards)
                return sum(sorted_rewards[:num_to_keep])

            metrics.append(compute_cvar)

        elif config['type'].lower() == 'wang_c':
            config.setdefault('eta', 0.5)
            eta_1 = config['eta']

            def wang_c(p):
                return norm.cdf(norm.ppf(p) + eta_1)

            def compute_wang_c(rewards):
                """  Estimates Wang metric based on rewards  """
                sorted_rewards = sorted(rewards)
                n = len(rewards)
                return sum([sorted_rewards[i] * (wang_c((i + 1) / n) - wang_c(i / n)) for i in range(n)])

            metrics.append(compute_wang_c)

        elif config['type'].lower() == 'wang_a':
            config.setdefault('eta', -0.5)
            eta_2 = config['eta']

            def wang_a(p):
                return norm.cdf(norm.ppf(p) + eta_2)

            def compute_wang_a(rewards):
                """  Estimates Wang metric based on rewards  """
                sorted_rewards = sorted(rewards)
                n = len(rewards)
                return sum([sorted_rewards[i] * (wang_a((i + 1) / n) - wang_a(i / n)) for i in range(n)])

            metrics.append(compute_wang_a)

        elif config['type'].lower() == 'pow_c':

            config.setdefault('eta', 2)
            eta_1 = config['eta']
            assert eta_1 >= 0, 'Need positive eta for pow_c metric'

            def pow_c(p):
                return p ** (1 / (1 + eta_1))

            def compute_pow_c(rewards):
                """  Estimates Pow metric based on rewards  """
                sorted_rewards = sorted(rewards)
                n = len(rewards)
                return sum([sorted_rewards[i] * (pow_c((i + 1) / n) - pow_c(i / n)) for i in range(n)])

            metrics.append(compute_pow_c)

        elif config['type'].lower() == 'pow_a':

            config.setdefault('eta', -2)
            eta_2 = config['eta']
            assert eta_2 < 0, 'Need negative eta for pow_a metric'

            def pow_a(p):
                return 1 - (1 - p) ** (1 / (1 - eta_2))

            def compute_pow_a(rewards):
                """  Estimates Pow metric based on rewards  """
                sorted_rewards = sorted(rewards)
                n = len(rewards)
                return sum([sorted_rewards[i] * (pow_a((i + 1) / n) - pow_a(i / n)) for i in range(n)])

            metrics.append(compute_pow_a)

        else:
            raise NotImplementedError('other evaluation metrics not implemented')
    return metrics
