import numpy as np
from cython.parallel import parallel, prange
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline float cosine_similarity(float[:] vector_a, float[:] vector_b) nogil:
    """Calculate cosine similarity between two vectors."""
    cdef float dot_product = 0.0
    cdef float norm_a = 0.0
    cdef float norm_b = 0.0
    cdef Py_ssize_t i
    for i in range(vector_a.shape[0]):
        dot_product += vector_a[i] * vector_b[i]
        norm_a += vector_a[i] ** 2
        norm_b += vector_b[i] ** 2
    return dot_product / (norm_a**0.5 * norm_b**0.5)


def swap_fitness(
    float[:,::1] population,
    float[:] population_fitness,
    int index,
    float[:,::1] population2,
    float[:] population2_fitness,
    int index2,
):
    """Swap positions between two populations."""
    cdef float[:] temp_swap = population[index].copy()
    cdef float temp_swap_fitness = population_fitness[index]
    population[index] = population2[index2]
    population_fitness[index] = population2_fitness[index2]
    population2[index2] = temp_swap
    population2_fitness[index2] = temp_swap_fitness


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function 
def best_generation(
    float[:,::1] population,
    float[:] population_fitness,
    float[:,::1] offspring_population,
    float[:] offspring_population_fitness,
    int n_population,
):
    """Get best generation.
    Get best population from population and offspring by fitness.
    Similat to tournament selection.

    """
    cdef int index
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




@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function 
def select(
    float[:,::1] population,
    float[:] population_fitness,
    float[:,::1] offspring_population,
    float[:] offspring_population_fitness,
    int n_population,
    float threshold,
    int num_threads
) -> np.ndarray:
    """Select individuals.
    Use shared fitness.
    The fitness is shared using cosine similarity.
    """
    cdef int num_population = population.shape[0]
    # Group individuals by cosine similarity
    cdef int total_population = num_population + offspring_population.shape[0]
    cdef float[:] num_members = np.zeros(total_population, dtype=np.float32)
    population[:, 1] = -100
    offspring_population[:, 1] = -100
    cdef Py_ssize_t index_start_ind = 2
    cdef float simil = 0.0
    # Fitness shared using cosine similarity
   
    cdef Py_ssize_t index, index2

    for index in prange(total_population, nogil=True, schedule='guided', num_threads=num_threads):
        # Calculate cosine similarity for population
        for index2 in range(total_population):
            # Population with itself
            if index < num_population and index2 < num_population:
                simil = cosine_similarity(
                    population[index, index_start_ind:],
                    population[index2, index_start_ind:],
                )
                if simil > threshold:
                    num_members[index] += simil
            # Population with offspring
            elif index < num_population and index2 >= num_population:
                simil = cosine_similarity(
                    population[index, index_start_ind:],
                    offspring_population[index2 - num_population, index_start_ind:],
                )
                if simil > threshold:
                    num_members[index] += simil
            # Offspring with offspring
            elif index >= num_population and index2 >= num_population:
                simil = cosine_similarity(
                    offspring_population[index - num_population, index_start_ind:],
                    offspring_population[index2 - num_population, index_start_ind:],
                )
                if simil > threshold:
                    num_members[index] += simil
            # Offspring with population
            elif index >= num_population and index2 < num_population:
                simil = cosine_similarity(
                    offspring_population[index - num_population, index_start_ind:],
                    population[index2, index_start_ind:],
                )
                if simil > threshold:
                    num_members[index] += simil

    for index in range(population.shape[0]):
        population_fitness[index] = population_fitness[index] / num_members[index]

    for index in range(offspring_population.shape[0]):
        offspring_population_fitness[index] = (
            offspring_population_fitness[index] / num_members[index + num_population]
        )

    best_generation(
        population,
        population_fitness,
        offspring_population,
        offspring_population_fitness,
        n_population,
    )




cdef extern from "evolutionary_operations.h":
    float perfomance_by_doc(
        float *individual,
        float **doc,
        int *target,
        int doc_size,
        int individual_size,
        int unknown_id,
        int entity_id
    )

    float fitness_by_individual(
        float *individual,
        float ***docs,
        int **targets,
        float **meta,
        int unknown_id,
        int docs_count
    )

    float *fitness(
        float **population,
        float ***docs,
        int **targets,
        float **meta,
        int unknown_id,
        int docs_count,
        int population_size,
        int n_threads
    )

from libc.stdlib cimport malloc, free

cpdef float cpp_perfomance_by_doc(
    float[:] individual,
    float[:,::1] doc,
    int[:] target,
    int doc_size,
    int individual_size,
    int unknown_id,
    int entity_id=1
):
    """Performance of individual on doc."""
    #cdef float *individual_data = <float *>individual.data
    #cdef int *target_data = <int *>target.data
    cdef float **doc_data = <float **>malloc(doc.shape[0] * sizeof(float *))
    cdef int i
    for i in range(doc.shape[0]):
        doc_data[i] = &doc[i, 0]


    return perfomance_by_doc(
        &individual[0],
        doc_data,
        &target[0],
        doc_size,
        individual_size,
        unknown_id,
        entity_id
    )

cpdef float cpp_fitness_by_individual(
    float[:] individual,
    float[:,:,::1] docs,
    int[:,::1] targets,
    float[:,::1] meta,
    int unknown_id,
    int docs_count
    ):
    """Fitness function.
    """

    cdef float ***docs_data = <float ***>malloc(docs_count * sizeof(float **))
    cdef int **targets_data = <int **>malloc(docs_count * sizeof(int *))
    cdef float **meta_data = <float **>malloc(docs_count * sizeof(float *))
    
    cdef int i, j
    for i in range(docs_count):
        docs_data[i] = <float **>malloc(docs.shape[1] * sizeof(float *))
        for j in range(docs.shape[1]):
            docs_data[i][j] = &docs[i, j, 0]
        targets_data[i] = &targets[i, 0]
        meta_data[i] = &meta[i, 0]
    
    return fitness_by_individual(
        &individual[0],
        docs_data,
        targets_data,
        meta_data,
        unknown_id,
        docs_count
    )

cimport openmp

cpdef float[:] cpp_fitness(
    float[:,::1] population,
    float[:,:,::1] docs,
    int[:,::1] targets,
    float[:,::1] meta,
    int unknown_id,
    int n_threads
    ):
    """Fitness function.
    """
    cdef population_size = population.shape[0]
    cdef docs_count = docs.shape[0]
    cdef float **population_data = <float **>malloc(population_size * sizeof(float *))
    cdef float ***docs_data = <float ***>malloc(docs_count * sizeof(float **))
    cdef int **targets_data = <int **>malloc(docs_count * sizeof(int *))
    cdef float **meta_data = <float **>malloc(docs_count * sizeof(float *))
    cdef int i, j
    for i in range(population_size):
        population_data[i] = &population[i, 0]

    for i in range(docs_count):
        docs_data[i] = <float **>malloc(docs.shape[1] * sizeof(float *))
        for j in range(docs.shape[1]):
            docs_data[i][j] = &docs[i, j, 0]
        targets_data[i] = &targets[i, 0]
        meta_data[i] = &meta[i, 0]

    cdef float *fitness_data = fitness(
        population_data,
        docs_data,
        targets_data,
        meta_data,
        unknown_id,
        docs_count,
        population_size,
        n_threads
    )
    cdef float[:] fitness_arr = np.zeros(population_size, dtype=np.float32)
    for i in range(population_size):
        fitness_arr[i] = fitness_data[i]
    return fitness_arr
