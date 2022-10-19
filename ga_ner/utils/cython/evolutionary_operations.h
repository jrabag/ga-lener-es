#ifndef EVOLUTIONARY_OPERATIONS_H
#define EVOLUTIONARY_OPERATIONS_H

float perfomance_by_doc(
    float *individual,
    float **doc,
    int *target,
    int doc_size,
    int individual_size,
    int unknown_id,
    int entity_id);

float fitness_by_individual(
    float *individual,
    float ***docs,
    int **targets,
    float **meta,
    int unknown_id,
    int docs_count);

float *fitness(
    float **population,
    float ***docs,
    int **targets,
    float **meta,
    int unknown_id,
    int docs_count,
    int population_size,
    int n_threads);

#endif