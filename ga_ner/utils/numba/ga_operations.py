from typing import Dict, List, Tuple

import numpy as np

from numba import njit

from .functions import np_any_axis1
from .similarity import cosine_similarity


# @njit
def slice_doc(
    doc: np.ndarray,
    target: np.ndarray,
    windows: int,
    doc_size: int,
    feature_array: np.ndarray,
    embedding_size: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate slice of doc.
    Return list of tuple with slice of doc and target.
    """
    # TODO test with windows > 1 and embedding_size > 1
    # TODO test with windows = 1 and embedding_size > 1
    slices: List[Tuple[np.ndarray, np.ndarray]] = []
    for index in range(0, doc_size - windows + 2):
        start_index = index * embedding_size * 3
        end_index = (index + windows) * embedding_size * 3
        doc_slice = np.zeros(windows * embedding_size, dtype=np.float32)
        for index_feature in range(feature_array.shape[0]):
            displacement = feature_array[index_feature].item() * embedding_size
            segment_start = index_feature * embedding_size
            # index_segment_doc = doc index + displacement feature + displacement rule
            index_segment_doc = (
                start_index + displacement + index_feature * embedding_size * 3
            )
            if index_segment_doc >= doc.shape[0]:
                print(f"{index} Doc size is too long ", doc_size)
                break
            if end_index < index_segment_doc + embedding_size:
                raise ValueError("Index out of range")
            doc_slice[segment_start : segment_start + embedding_size] = doc[
                index_segment_doc : index_segment_doc + embedding_size
            ]
        slices.append((doc_slice, target[index : index + windows]))
        # doc_slice[:] = doc[start_index:end_index].reshape(3, -1)
        # slices.append((doc_slice, target[index : index + windows]))
    return slices


# @njit(cache=True)
def perfomance_by_doc(
    individual: np.ndarray,
    doc: np.ndarray,
    target: np.ndarray,
    doc_size: int,
    individual_size: int,
    *,
    entity_id: int,
    embedding_size: int = 1,
    num_features: int = 3,
    similarity_threshold: float = 0.8,
    entity_label: int | None = None,
):
    """Performance of individual on doc.

    entity_label: (int|None) - label of entity which island is searching.
    """

    # Validate if target is has entity label
    if doc_size == 0:
        return 0, 0, 0
    # if entity_label is not None:
    #     # DOC
    #     if not np.any(target > 0):
    #         return -1.0

    if individual_size > doc_size + 2:
        return 0, 0, 0

    # individual_target = target == entity_id
    if not np.any(target > 0):
        return 0, 0, 0

    individual_data = np.zeros(individual_size * embedding_size, dtype=np.float32)
    entity_mask = np.zeros(individual_size, dtype=np.bool8)
    entities_rule = np.zeros(individual_size, dtype=np.int32)
    feature_array = np.zeros(individual_size, dtype=np.int32)
    # Fill individual data, ignore first 2 elements for each segment
    for index in range(individual_size):
        start_index = index * (embedding_size + 2) + 2
        end_index = start_index + embedding_size
        individual_data[
            index * embedding_size : (index + 1) * embedding_size
        ] = individual[start_index:end_index]
        # Set mask of entity
        entity_mask[index] = bool(individual[start_index - 2])
        entities_rule[index] = int(individual[start_index - 2])
        # Set type of feature (0 - POS, 1 - Dep, 2 - Word)
        feature_array[index] = int(individual[start_index - 1])

    union_doc: int = 0
    intercep_doc: int = 0
    retrive_doc: int = 0
    total_matches: int = 0
    similarity_accumulator: float = 0.0

    for sliced_doc, sliced_targed in slice_doc(
        doc, target, individual_size, doc_size, feature_array, embedding_size
    ):
        # TODO Validate individual_data
        # Cosine similarity for each segment
        similarity = 0
        for i in range(individual_size):
            try:
                similarity += cosine_similarity(
                    sliced_doc[i * embedding_size : (i + 1) * embedding_size],
                    individual_data[i * embedding_size : (i + 1) * embedding_size],
                )
            except ZeroDivisionError:
                raise
        similarity /= individual_size
        # similarity = sliced_doc @ individual_data
        # similarity /= np.linalg.norm(sliced_doc) * np.linalg.norm(individual_data)

        predict_span: np.ndarray = np.zeros(individual_size, dtype=np.bool8)

        if similarity > 1 + 1e-6:
            raise ValueError("Similarity > 1")

        if similarity > similarity_threshold:
            similarity_accumulator += similarity
            total_matches += 1
            for i in range(individual_size):
                predict_span[i] = bool(entity_mask[i])

            intercep_doc += (
                (predict_span & (sliced_targed == entities_rule)).sum().item()
            )
            retrive_doc += predict_span.sum().item()
            union_doc += (predict_span | (sliced_targed == entities_rule)).sum().item()

        if intercep_doc > retrive_doc:
            print(f"{intercep_doc} {retrive_doc}")

    # if union_doc == 0:
    #     return -1.0

    # if total_matches == 0:
    #     return 0.0

    # if intercep_doc == 0:
    #     return 0.0

    # if retrive_doc == 0:
    #     # return (
    #     #     0.1 * np.log2(individual_size) / np.log2(7)
    #     #     + 0.1 * similarity_accumulator / total_matches
    #     # )
    #     return -1

    # if intercep_doc / retrive_doc > 1:
    #     print(intercep_doc / retrive_doc, intercep_doc, retrive_doc)

    # return (
    #     0.8 * (intercep_doc / retrive_doc)
    #     + 0.1 * np.log2(individual_size) / np.log2(7)
    #     + 0.1 * similarity_accumulator / total_matches
    # )
    return intercep_doc, union_doc, retrive_doc  # * np.log2(intercep_doc)


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
        if population_fitness[index] < offspring_population_fitness[index * 2]:
            swap_fitness(
                population,
                population_fitness,
                index,
                offspring_population,
                offspring_population_fitness,
                index * 2,
            )
        if population_fitness[index] < offspring_population_fitness[index * 2 + 1]:
            swap_fitness(
                population,
                population_fitness,
                index,
                offspring_population,
                offspring_population_fitness,
                index * 2 + 1,
            )


def select(
    population: np.ndarray,
    population_fitness: np.ndarray,
    offspring_population: np.ndarray,
    offspring_population_fitness: np.ndarray,
    n_population: int,
    *,
    threshold: float,
    num_threads: int = 0,
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
    sharing: np.ndarray = np.zeros(
        population.shape[0] + offspring_population.shape[0], dtype=np.float64
    )
    population[:, 1] = -100
    offspring_population[:, 1] = -100
    # Get index where embedding is in
    embedding_size = 32
    max_len = 241
    embedding_list = []
    for index in range(3, max_len, embedding_size + 2):
        embedding_list.extend(range(index + 2, index + embedding_size + 2))
    # Fitness shared using cosine similarity
    index_start_ind = 3
    population2 = population[:, embedding_list]
    offspring_population2 = offspring_population[:, embedding_list]
    for index in range(population.shape[0]):
        for index2 in range(0, population.shape[0]):
            simil = cosine_similarity(
                population2[index],
                population2[index2],
            )
            if simil >= threshold:
                sharing[index] += 1 - ((1 - simil) / (1 - threshold))

        for index2 in range(0, offspring_population.shape[0]):
            simil = cosine_similarity(
                population2[index],
                offspring_population2[index2],
            )
            if simil >= threshold:
                sharing[index] += 1 - ((1 - simil) / (1 - threshold))

    offset = population.shape[0]
    for index in range(offspring_population.shape[0]):
        for index2 in range(offspring_population.shape[0]):
            simil = cosine_similarity(
                offspring_population2[index],
                offspring_population2[index2],
            )
            if simil >= threshold:
                sharing[index + offset] += 1 - ((1 - simil) / (1 - threshold))

        for index2 in range(population.shape[0]):
            simil = cosine_similarity(
                offspring_population2[index],
                population2[index2],
            )
            if simil >= threshold:
                sharing[index + offset] += 1 - ((1 - simil) / (1 - threshold))

    population_fitness[:] = population_fitness[:] / sharing[: population.shape[0]]
    offspring_population_fitness[:] = (
        offspring_population_fitness[:] / sharing[population.shape[0] :]
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
    *,
    entity_id: int,
    num_features: int,
    embedding_size: int,
) -> float:
    """Fitness function.
    Return fitness of individual.
    F1 score
    F(R) = frac{2*S_p*S_r,S_p + S_r)
    """
    unions: np.ndarray = np.zeros(len(docs))
    tps: np.ndarray = np.zeros(len(docs))
    retrieves: np.ndarray = np.zeros(len(docs))
    individual_size: int = int(individual[0])
    indivual_rep = individual[3:]
    # TODO add entity_label
    for index_doc, doc in enumerate(docs):
        tp, union, retrieve = perfomance_by_doc(
            indivual_rep,
            doc,
            targets[index_doc],
            int(meta[index_doc, 0]),
            individual_size,
            entity_id=entity_id,
            num_features=num_features,
            embedding_size=embedding_size,
        )
        # perfomance_doc[index_doc] = perfomance
        document_confidence = meta[index_doc, 2] / 100.0
        unions[index_doc] = union
        tps[index_doc] = tp * document_confidence
        retrieves[index_doc] = retrieve

    union = unions.sum()
    tp = tps.sum()
    retrieve = retrieves.sum()
    if retrieve == 0:
        individual[2] = 0
    else:
        individual[2] = tp / retrieves.sum()
    # individual[2] = retrieves.sum()
    if union == 0:
        return 0
    if tp == 0:
        return 0
    if retrieve == 0:
        return 0
    # return individual[2]
    return tp / retrieve * np.log2(tp)
    # if perfomance_doc[perfomance_doc >= 0].mean() > 1:
    #     print(perfomance_doc[perfomance_doc >= 0].mean())
    # filter_perfomance_doc = perfomance_doc >= 0
    # if filter_perfomance_doc.shape[0] == 0:
    #     return 0
    # perfomance_acc = perfomance_doc[filter_perfomance_doc].sum()
    # if perfomance_acc == 0:
    #     return 0
    # return perfomance_acc / filter_perfomance_doc.shape[0]


def fitness(
    input_data,
    target,
    meta,
    population: np.ndarray,
    num_features: int,
    embedding_size: int,
) -> np.ndarray:
    """Fitness function.
    Return fitness of population.
    F1 score
    F(R) = frac{2*S_p*S_r,S_p + S_r)
    """
    perfomance_doc = np.zeros(population.shape[0], dtype=np.float32)
    for index, individual in enumerate(population):
        limit_feature: int = 0

        perfomance_doc[index] = fitness_by_individual(
            individual,
            input_data,
            target,
            meta,
            entity_id=int(individual[2].item()),
            num_features=num_features,
            embedding_size=embedding_size,
        )

    return perfomance_doc
