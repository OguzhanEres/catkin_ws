#!/usr/bin/env python3
"""
Recurrent minibatch iterator for PPO+LSTM.
"""
import numpy as np


def recurrent_minibatches(data, seq_len, batch_size):
    """
    data: dict of np arrays with leading dim T
    Yields index arrays to slice sequences while roughly respecting order.
    """
    T = next(iter(data.values())).shape[0]
    idx = np.arange(T)
    np.random.shuffle(idx)
    chunks = [idx[i : i + seq_len] for i in range(0, T, seq_len)]
    np.random.shuffle(chunks)
    batch = []
    total = 0
    for chunk in chunks:
        batch.append(chunk)
        total += len(chunk)
        if total >= batch_size:
            yield np.concatenate(batch)
            batch = []
            total = 0
    if batch:
        yield np.concatenate(batch)
