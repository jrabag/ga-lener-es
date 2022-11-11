#include <cuda.h>
#include <cuda_runtime.h>

#ifndef EVOLUTIONARY_OPERATIONS_CUH
#define EVOLUTIONARY_OPERATIONS_CUH

__device__ float perfomance_by_doc(
    float *individual,
    float **doc,
    int *target,
    int doc_size,
    int individual_size,
    int unknown_id,
    int entity_id);

__device__ float fitness_by_individual(
    float *individual,
    float ***docs,
    int **targets,
    float **meta,
    int unknown_id,
    int docs_count);

__device__ float *fitness(
    float **population,
    float ***docs,
    int **targets,
    float **meta,
    int unknown_id,
    int docs_count,
    int population_size,
    int n_threads);

#endif