from numba import njit
import numpy as np
from .similarity import cosine_similarity


@njit(cache=True)
def np_any_axis1(x):
    """Numba compatible version of np.any(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out


@njit(cache=True)
def np_sum_axis1(x):
    """Numba compatible version of np.sum(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=np.int64)
    for i in range(x.shape[1]):
        out = np.sum(out, x[:, i])
    return out


@njit(cache=True)
def calculate_shared_fitness(population, fitness, sigma=0.1):
    """Calculate shared fitness for a population.
    
    $$d(a, b) = 1- \frac{a \cdot b}{\|a\| \|b\|}$$

    $$
    sh(d) = \left \{
    \begin{array}{l}
        1  - \frac{d}{\sigma_{share}}, d < {\sigma_{share}} \\
        0, otherwise
    \end{array}
    \right \}
    $$


    $$
    f^t_i = \frac{f_i}{\sum_{j=1}^N sh(d_{i,j})}
    $$
    Args:
        population (np.ndarray): Population of individuals.
        fitness (np.ndarray): Fitness values of individuals.
        sigma (float, optional): Sharing distance. Defaults to 0.1. 
    Returns:
        np.ndarray: Shared fitness values of individuals.

    """
    # Calculate similarity matrix
    distance_matrix = np.zeros((population.shape[0], population.shape[0]))
    for i in range(population.shape[0]):
        for j in range(population.shape[0]):
            distance_matrix[i, j] = 1 - cosine_similarity(population[i], population[j])
    # Calculate shared fitness
    shared_fitness = np.zeros(fitness.shape)
    for i in range(fitness.shape[0]):
        shared_fitness[i] = fitness[i] / np.sum(
            np.where(distance_matrix[i] < sigma, 1 - distance_matrix[i] / sigma, 0)
        )
    return shared_fitness
