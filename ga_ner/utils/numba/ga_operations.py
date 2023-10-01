from typing import Dict, List, Tuple

import numpy as np

from numba import njit

from .functions import np_any_axis1
from .similarity import cosine_similarity


@njit
def slice_doc(
    doc: np.ndarray, target: np.ndarray, windows: int, doc_size
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate slice of doc."""
    slices: List[Tuple[np.ndarray, np.ndarray]] = []
    for index in range(0, doc_size - windows + 2):
        slices.append((doc[index : index + windows], target[index : index + windows]))
    return slices


@njit(cache=True)
def perfomance_by_doc(
    individual: np.ndarray,
    doc: np.ndarray,
    target: np.ndarray,
    doc_size: int,
    individual_size: int,
    unknown_id: int,
):
    """Performance of individual on doc."""

    individual_data = individual[:individual_size].reshape(-1, 1)
    entity_mask: np.ndarray = individual_data > 0
    mask_unknown: np.ndarray = individual_data == unknown_id
    union_doc: int = 0
    intercep_doc: int = 0
    retrive_doc: int = 0

    if individual_size > doc_size + 2:
        return 0.0

    for sliced_doc, sliced_targed in slice_doc(doc, target, individual_size, doc_size):
        macth_tokens: np.ndarray = (sliced_doc == np.abs(individual_data)) | (
            mask_unknown
        )

        any_match = np_any_axis1(macth_tokens)
        total_match = np.sum(any_match)

        predict_span: np.ndarray = np.zeros(individual_size, dtype=np.bool8)
        if total_match == individual_size:
            # predict_span: np.ndarray = np_sum_axis1(macth_tokens * entity_mask)
            # np_sum_axis1(macth_tokens * entity_mask)
            for i in range(individual_size):
                predict_span[i] = np.sum(macth_tokens[i] * entity_mask[i])

        intercep_doc += (predict_span * sliced_targed.flatten()).sum()
        union_doc += (predict_span | sliced_targed.flatten()).sum()
        retrive_doc += predict_span.sum()
        if intercep_doc > retrive_doc:
            print(f"{intercep_doc} {retrive_doc}")

    if union_doc == 0:
        return -1.0

    if intercep_doc == 0:
        return 0.0

    if intercep_doc / retrive_doc > 1:
        print(intercep_doc / retrive_doc, intercep_doc, retrive_doc)

    return intercep_doc / retrive_doc


def swap_fitness(
    population: np.ndarray,
    population_fitness: np.ndarray,
    index: int,
    population2: np.ndarray,
    population2_fitness: np.ndarray,
    index2: int,
):
    """Swap postions of population and population2"""
    temp_swap = population[index].copy()
    temp_swap_fitness = population_fitness[index]
    population[index] = population2[index2].copy()
    population_fitness[index] = population2_fitness[index2]
    population2[index2] = temp_swap
    population2_fitness[index2] = temp_swap_fitness


def best_generation(
    population: np.ndarray,
    population_fitness: np.ndarray,
    offspring_population: np.ndarray,
    offspring_population_fitness: np.ndarray,
    n_population: int,
):
    """Get best generation.
    Get best population from population and offspring by fitness.
    Similar to tournament selection.

    """
    for index in range(n_population):
        # Horizontal swap
        if index + 1 < n_population:
            if population_fitness[index] < population_fitness[index + 1]:
                swap_fitness(
                    population,
                    population_fitness,
                    index,
                    population,
                    population_fitness,
                    index + 1,
                )
            if (
                offspring_population_fitness[index]
                > offspring_population_fitness[index + 1]
            ):
                swap_fitness(
                    offspring_population,
                    offspring_population_fitness,
                    index,
                    offspring_population,
                    offspring_population_fitness,
                    index + 1,
                )
            if (
                offspring_population_fitness[index + n_population]
                > offspring_population_fitness[index + n_population + 1]
            ):
                swap_fitness(
                    offspring_population,
                    offspring_population_fitness,
                    index + n_population,
                    offspring_population,
                    offspring_population_fitness,
                    index + n_population + 1,
                )
        # Vertical swap
        if population_fitness[index] < offspring_population_fitness[index]:
            swap_fitness(
                population,
                population_fitness,
                index,
                offspring_population,
                offspring_population_fitness,
                index,
            )
        if (
            population_fitness[index]
            < offspring_population_fitness[index + n_population]
        ):
            swap_fitness(
                population,
                population_fitness,
                index,
                offspring_population,
                offspring_population_fitness,
                index + n_population,
            )


def select(
    population: np.ndarray,
    population_fitness: np.ndarray,
    offspring_population: np.ndarray,
    offspring_population_fitness: np.ndarray,
    n_population: int,
    *,
    threshold: float,
) -> np.ndarray:
    """Select individuals.
    1. Get best individuals from population and offspring.
    2. Group individuals by cosine similarity
        1. If cosine similarity is higher than {threshold}, add to group
        2. If cosine similarity is lower than {threshold}, create new group
        3. if a group has less than {min_individual} individuals, add to most similar group
    3. Select best individuals from each group
    """

    # Group individuals by cosine similarity
    num_members: np.ndarray = np.zeros(
        population.shape[0] + offspring_population.shape[0], dtype=np.float64
    )
    population[:, 1] = -100
    offspring_population[:, 1] = -100
    # Fitness shared using cosine similarity
    index_start_ind = 2
    for index in range(population.shape[0]):
        for index2 in range(0, population.shape[0]):
            simil = cosine_similarity(
                population[index, index_start_ind:],
                population[index2, index_start_ind:],
            )
            if simil > threshold:
                num_members[index] += simil

        for index2 in range(0, offspring_population.shape[0]):
            simil = cosine_similarity(
                population[index, index_start_ind:],
                offspring_population[index2, index_start_ind:],
            )
            if simil > threshold:
                num_members[index] += simil

    offset = population.shape[0]
    for index in range(offspring_population.shape[0]):
        for index2 in range(offspring_population.shape[0]):
            simil = cosine_similarity(
                offspring_population[index, index_start_ind:],
                offspring_population[index2, index_start_ind:],
            )
            if simil > threshold:
                num_members[index + offset] += simil

        for index2 in range(population.shape[0]):
            simil = cosine_similarity(
                offspring_population[index, index_start_ind:],
                population[index2, index_start_ind:],
            )
            if simil > threshold:
                num_members[index + offset] += simil

    population_fitness[:] = population_fitness[:] / num_members[: population.shape[0]]
    offspring_population_fitness[:] = (
        offspring_population_fitness[:] / num_members[population.shape[0] :]
    )

    best_generation(
        population,
        population_fitness,
        offspring_population,
        offspring_population_fitness,
        n_population,
    )


def fitness_by_individual(
    individual: np.ndarray,
    docs: np.ndarray,
    targets: np.ndarray,
    meta: np.ndarray,
    unknown_id: int,
) -> float:
    """Fitness function.
    Return fitness of individual.
    F1 score
    F(R) = frac{2*S_p*S_r,S_p + S_r)
    """
    perfomance_doc: np.ndarray = np.zeros(len(docs))
    individual_size: int = int(individual[0])
    indivual_rep = individual[3:]

    for index_doc, doc in enumerate(docs):
        perfomance = perfomance_by_doc(
            indivual_rep,
            doc,
            targets[index_doc],
            int(meta[index_doc, 0]),
            individual_size,
            unknown_id,
        )
        perfomance_doc[index_doc] = perfomance

    # if perfomance_doc[perfomance_doc >= 0].mean() > 1:
    #     print(perfomance_doc[perfomance_doc >= 0].mean())
    return perfomance_doc[perfomance_doc >= 0].mean()


def fitness(
    population: np.ndarray, data: Dict, map_inv_entity: Dict[int, str], unknown_id: int
) -> np.ndarray:
    """Fitness function.
    Return fitness of population.
    F1 score
    F(R) = frac{2*S_p*S_r,S_p + S_r)
    """
    perfomance_doc = np.zeros(len(population.shape[0]), dtype=np.float32)
    for index, individual in enumerate(population):
        entity_type = map_inv_entity[individual[2]]
        perfomance_doc[index] = fitness_by_individual(
            individual,
            data[entity_type]["input"],
            data[entity_type]["target"],
            data[entity_type]["meta"],
            unknown_id,
        )

    return perfomance_doc
