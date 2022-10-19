from numba import njit
import numpy as np


@njit(cache=True)
def cosine_similarity(vector1, vector2):
    """Calculate cosine similarity between two vectors."""
    return (vector1 * vector2).sum() / (
        np.sqrt((vector1 * vector1).sum()) * np.sqrt((vector2 * vector2).sum())
    )
