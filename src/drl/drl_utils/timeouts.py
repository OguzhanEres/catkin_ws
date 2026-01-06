#!/usr/bin/env python3
"""
Timeout and retry helpers.
"""
import time


def wait_until(cond_fn, timeout, sleep=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond_fn():
            return True
        time.sleep(sleep)
    return False


def call_with_retry(fn, retries=3, sleep=0.5):
    for i in range(retries):
        try:
            return fn()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(sleep)
