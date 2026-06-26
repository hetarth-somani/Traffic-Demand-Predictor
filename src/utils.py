"""
Utility functions: timers, logging, mathematical clips.
"""
import time
import numpy as np

def clip_preds(preds: np.ndarray, max_val: float) -> np.ndarray:
    """
    Clip predictions to be strictly non-negative, and cap at a max value + 10%.
    """
    return np.clip(preds, 0.0, max_val * 1.10)

class Timer:
    """Simple context manager for timing code blocks."""
    def __init__(self, name="Task"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        print(f"[{self.name}] started...")

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"[{self.name}] finished in {elapsed:.2f}s.")
