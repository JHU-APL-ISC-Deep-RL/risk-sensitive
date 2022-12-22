import numpy as np


def load_utility_function(config: dict):
    """
    Returns object for converting episode rewards into utilities.  Can either compute one utility for the whole
    episode or per step utilities (based on diff of full-episode utility at each time step).  Computes baseline
    (utility value at t=0) for logging purposes.
    """
    config.setdefault('type', 'identity')

    if config['type'].lower() == 'identity':  # utility = reward
        config['baseline'] = 0

        def compute_utility(rewards):
            return rewards

    elif config['type'].lower() == 'cpt':  # configurable CPT utility
        config.setdefault('reference', 0)
        config.setdefault('lambda', 2.25)
        config.setdefault('sigma', 0.88)
        ref, lam, sig = config['reference'], config['lambda'], config['sigma']
        config['baseline'] = ((0 >= ref) - lam * (0 < ref)) * np.abs(ref)**sig

        def compute_utility(rewards, ref1=None):
            cum_rewards = np.cumsum(np.concatenate(([0], rewards)))
            if ref1 is not None:
                cum_utilities = ((cum_rewards >= ref1) - lam * (cum_rewards < ref1)) * np.abs(cum_rewards - ref1) ** sig
            else:
                cum_utilities = ((cum_rewards >= ref) - lam * (cum_rewards < ref)) * np.abs(cum_rewards - ref) ** sig
            per_step_utilities = np.diff(cum_utilities)
            return per_step_utilities

    elif config['type'].lower() == 'crowdnav':  # variable reference CPT utility for CrowdNav
        config.setdefault('progress_reference', 0.5)
        config.setdefault('time_factor', 0.3)
        config.setdefault('lambda', 2.25)
        config.setdefault('sigma', 0.88)
        progress_ref, time_factor = config['progress_reference'], config['time_factor']
        lam, sig = config['lambda'], config['sigma']
        config['baseline'] = ((0 >= progress_ref) - lam * (0 < progress_ref)) * np.abs(ep_reward - progress_ref) ** sig

        def compute_utility(rewards, progress):  # requires per-step progress in addition to rewards
            cum_rewards = np.cumsum(np.concatenate(([0], rewards)))
            refs = progress_ref*np.ones(rewards.shape) - time_factor*np.cumsum(progress)
            refs = np.concatenate(([progress_ref], refs))
            cum_utilities = ((cum_rewards >= refs) - lam * (cum_rewards < refs)) * np.abs(cum_rewards - refs) ** sig
            per_step_utilities = np.diff(cum_utilities)
            return per_step_utilities

    elif config['type'].lower() == 'linear':  # configurable; different slopes on either side of a reference
        config.setdefault('reference', 0)
        config.setdefault('positive_slope', 1)
        config.setdefault('negative_slope', 1)
        ref, m_plus, m_minus = config['reference'], config['positive_slope'], config['negative_slope']
        config['baseline'] = ((0 >= ref)*m_plus + (0 < ref)*m_minus)*(-ref)

        def compute_utility(rewards, ref1=None):
            cum_rewards = np.cumsum(np.concatenate(([0], rewards)))
            if ref1 is not None:
                cum_utilities = ((cum_rewards >= ref1) * m_plus + (cum_rewards < ref1) * m_minus) * (cum_rewards - ref1)
            else:
                cum_utilities = ((cum_rewards >= ref) * m_plus + (cum_rewards < ref) * m_minus) * (cum_rewards - ref)
            per_step_utilities = np.diff(cum_utilities)
            return per_step_utilities

    else:
        raise NotImplementedError('Utility function not yet implemented')
    return compute_utility
