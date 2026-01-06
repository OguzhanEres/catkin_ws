#!/usr/bin/env python3
"""
GAE-Lambda helper for PPO.
"""
import numpy as np


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """Compute advantage and returns with GAE-Lambda."""
    T = len(rewards)
    adv = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns
