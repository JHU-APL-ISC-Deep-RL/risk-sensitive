import numpy as np
from scipy.stats import norm


def load_weight_coefficient_function(config, weight_function=None):
    """
    Returns function for computing weight coefficients for collected trajectories.  In the CDF-based policy gradient
    methods these coefficients are functions of dw/dP, where w is the prescribed weight function of the CDF P.  In
    this implementation, the derivative is estimated numerically.
    """
    config.setdefault('type', 'uniform')
    eps = 1e-5

    def compute_derivative(function, p):
        """  Computes 2-sided numerical estimate of derivative.  Assumes p +/- eps is in domain of function.  """
        return (function(p + eps) - function(p - eps)) / 2 / eps

    def get_sorted_indices(input_list):
        """  Sort a list in ascending order, returning a list of indices"""
        return [item[0] for item in sorted(enumerate(input_list), key=lambda x: x[1])]

    if weight_function is not None:
        if 'num_trajectories' in config:
            default_num_trajectories = config['num_trajectories']
            default_p = np.clip(np.array([i / default_num_trajectories
                                          for i in range(default_num_trajectories + 1)]), eps, 1 - eps)
            default_dw_dp = compute_derivative(config['weight'], default_p)
        else:
            default_num_trajectories, default_dw_dp = -1, []

        def compute_weight_coefficients(rewards):
            num_trajectories = len(rewards)
            if num_trajectories == default_num_trajectories:  # don't compute more often than we need to
                dw_dp = default_dw_dp
            else:
                p = np.clip(np.array([i / num_trajectories for i in range(num_trajectories + 1)]), eps, 1 - eps)
                dw_dp = compute_derivative(config['weight'], p)
            sorted_indices = get_sorted_indices(rewards)
            episode_weights = np.zeros((num_trajectories,))
            for i in range(len(rewards)):
                episode_weights[[sorted_indices[i]]] = dw_dp[i] + dw_dp[i + 1]
            return episode_weights * num_trajectories / np.sum(episode_weights)

    elif config['type'].lower() == 'uniform':  # all trajectories weighted the same

        def compute_weight_coefficients(rewards):
            return np.ones(len(rewards))

    elif config['type'].lower() == 'cpt':         # configurable CPT trajectory weighting (based on Tversky & Kahneman)
        config.setdefault('ep', 0.61)             # \eta_+
        config.setdefault('em', 0.69)             # \eta_-
        config.setdefault('reference', 0)         # reference
        config.setdefault('variable_ref', False)  # whether or not to use variable reference
        config.setdefault('tail', 0.03)           # region where weight functions are linearized (finite derivatives)
        ep, em, ref0 = config['ep'], config['em'], config['reference']
        v_ref, tail = config['variable_ref'], config['tail']

        def weight_plus(p):
            return (p < tail) * tail ** ep / (tail ** ep + (1 - tail) ** ep) ** (1 / ep) * p / tail + \
                   (p > 1 - tail) * ((1 - ((1 - tail) ** ep / ((1 - tail) ** ep + tail ** ep) ** (1 / ep))) * (
                    p - 1 + tail) / tail + ((1 - tail) ** ep / ((1 - tail) ** ep + tail ** ep) ** (1 / ep))) + \
                   (tail <= p) * (p <= 1 - tail) * (p ** ep / (p ** ep + (1 - p) ** ep) ** (1 / ep))

        def weight_minus(p):
            return (p < tail) * tail ** em / (tail ** em + (1 - tail) ** em) ** (1 / em) * p / tail + \
                   (p > 1 - tail) * ((1 - ((1 - tail) ** em / ((1 - tail) ** em + tail ** em) ** (1 / em))) * (
                    p - 1 + tail) / tail + ((1 - tail) ** em / ((1 - tail) ** em + tail ** em) ** (1 / em))) + \
                   (tail <= p) * (p <= 1 - tail) * (p ** em / (p ** em + (1 - p) ** em) ** (1 / em))

        if 'num_trajectories' in config:
            default_num_trajectories = config['num_trajectories']
            default_p = np.clip(np.array([i / default_num_trajectories
                                          for i in range(default_num_trajectories + 1)]), eps, 1 - eps)
            default_dwp_dp = compute_derivative(weight_plus, default_p)
            default_dwm_dp = compute_derivative(weight_minus, default_p)
        else:
            default_num_trajectories, default_dwp_dp, default_wmm = -1, [], []

        def compute_weight_coefficients(rewards):
            ref = np.mean(rewards) if v_ref else ref0
            num_trajectories = len(rewards)
            if num_trajectories == default_num_trajectories:  # don't compute more often than we need to
                dwp_dp = default_dwp_dp
                dwm_dp = default_dwm_dp
            else:
                p = np.clip(np.array([i / num_trajectories
                                      for i in range(num_trajectories + 1)]), eps, 1-eps)
                dwp_dp = compute_derivative(weight_plus, p)
                dwm_dp = compute_derivative(weight_minus, p)
            sorted_indices = get_sorted_indices(rewards)
            episode_weights = np.zeros((num_trajectories,))
            for i, reward in enumerate(rewards):
                j = sorted_indices.index(i)
                if reward >= ref:
                    episode_weights[i] = dwp_dp[-j - 1] + dwp_dp[-j - 2]
                else:
                    episode_weights[i] = dwm_dp[j] + dwm_dp[j + 1]
            return episode_weights * num_trajectories / np.sum(episode_weights)

    elif config['type'].lower() == 'wang' or config['type'].lower() == 'pow':

        if config['type'].lower() == 'wang':
            config.setdefault('eta', 0.75)  # risk-averse for eta > 0, risk-seeking for eta < 0
            config.setdefault('tail', 0.03)
            eta, tail = config['eta'], config['tail']

            def weight(p):
                if eta >= 0:
                    return (p < tail) * (norm.cdf(norm.ppf(tail) + eta) * p / tail) + \
                           (p >= tail) * (norm.cdf(norm.ppf(p) + eta))
                else:
                    return (p > 1 - tail) * ((1 - norm.cdf(norm.ppf(1-tail) + eta)) * (p - (1 - tail)) / tail) + \
                           (p <= 1 - tail) * (norm.cdf(norm.ppf(p) + eta))

        else:
            config.setdefault('eta', 2)
            config.setdefault('tail', 0.03)
            eta, tail = config['eta'], config['tail']

            def weight(p):
                if eta >= 0:
                    return (p < tail) * (tail ** (1 / (1 + eta)) * p / tail) + (p >= tail) * (p ** (1 / (1 + eta)))
                else:
                    return (p > 1 - tail) * (tail ** (1 / (1 - eta)) * (p - (1 - tail)) / tail) + \
                           (p <= 1 - tail) * (1 - (1 - p) ** (1 / (1 - eta)))

        if 'num_trajectories' in config:
            default_num_trajectories = config['num_trajectories']
            default_p = np.clip(np.array([i / default_num_trajectories
                                          for i in range(default_num_trajectories + 1)]), eps, 1 - eps)
            default_dw_dp = compute_derivative(weight, default_p)
        else:
            default_num_trajectories, default_dw_dp = -1, []

        def compute_weight_coefficients(rewards):
            num_trajectories = len(rewards)
            if num_trajectories == default_num_trajectories:  # don't compute more often than we need to
                dw_dp = default_dw_dp
            else:
                p = np.clip(np.array([i / num_trajectories for i in range(num_trajectories + 1)]), eps, 1 - eps)
                dw_dp = compute_derivative(weight, p)
            sorted_indices = get_sorted_indices(rewards)
            episode_weights = np.zeros((num_trajectories,))
            for i in range(len(rewards)):
                episode_weights[[sorted_indices[i]]] = dw_dp[i] + dw_dp[i + 1]
            return episode_weights * num_trajectories / np.sum(episode_weights)

    elif config['type'].lower() == 'cvar':  # configurable CVaR weighting; need to increase data collection for this
        config.setdefault('alpha', 0.05)  # fraction of trajectories to consider
        alpha = config['alpha']

        def compute_weight_coefficients(rewards):
            num_trajectories = len(rewards)
            num_to_keep = int(num_trajectories*alpha)
            remainder = num_trajectories*alpha - num_to_keep
            num_to_keep += int(np.random.rand() <= remainder)
            sorted_indices = get_sorted_indices(rewards)
            episode_weights = np.zeros((num_trajectories,))
            for i, reward in enumerate(rewards):
                j = sorted_indices.index(i)
                if j <= num_to_keep:
                    episode_weights[i] = 1
            return episode_weights * num_to_keep / np.sum(episode_weights)

    else:
        raise NotImplementedError('Weight function type is unknown.')
    return compute_weight_coefficients
