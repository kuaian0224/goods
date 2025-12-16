import numpy as np


def ensemble_mean(list_of_probs) -> np.ndarray:
    stacked = np.stack(list_of_probs, axis=0)
    return stacked.mean(axis=0)
