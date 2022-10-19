#include "evolutionary_operations.h"
#include "omp.h"
#include <stdio.h>
#include <math.h>

float perfomance_by_doc(
    float *individual,
    float **doc,
    int *target,
    int doc_size,
    int individual_size,
    int unknown_id,
    int entity_id)
{

    if (individual_size > doc_size + 2)
    {
        return 0.0;
    }

    int unionDoc = 0;
    int intercepDoc = 0;
    int retriveDoc = 0;
    bool entityDoc[doc_size];
    bool entityMask[individual_size];
    for (int i = 0; i < individual_size; i++)
    {
        entityMask[i] = individual[i] > 0;
    }
    for (int i = 0; i < doc_size; i++)
    {
        entityDoc[i] = false;
    }
    // Slice doc
    for (int indexDoc = 0; indexDoc < doc_size - individual_size + 2; indexDoc++)
    {
        // Slice tokens
        bool isMatch = true;
        for (int indexIndividual = 0; indexIndividual < individual_size; indexIndividual++)
        {
            bool anyMatch = false;
            for (int indexFeature = 0; indexFeature < 3; indexFeature++)
            {
                int token = doc[indexDoc + indexIndividual][indexFeature];
                int IndividualFeature = individual[indexIndividual];
                anyMatch = anyMatch || token == abs(IndividualFeature) || IndividualFeature == unknown_id;
            }
            isMatch = isMatch && anyMatch;
        }
        //  Evaluate match with target
        if (isMatch)
        {
            for (int indexIndividual = 0; indexIndividual < individual_size; indexIndividual++)
            {
                entityDoc[indexDoc + indexIndividual] = entityMask[indexIndividual];
            }
        }
    }

    for (int i = 0; i < doc_size + 2; i++)
    {
        bool targetToken = target[i] == entity_id;
        bool retriveToken = entityDoc[i];
        unionDoc += int(targetToken || retriveToken);
        intercepDoc += int(targetToken && retriveToken);
        retriveDoc += int(retriveToken);
    }

    if (unionDoc == 0)
    {
        return -1.0;
    }

    if (intercepDoc == 0)
    {
        return 0.0;
    }

    if (retriveDoc == 0)
    {
        return 0.0;
    }

    return (float)intercepDoc / retriveDoc;
}

float fitness_by_individual(
    float *individual,
    float ***docs,
    int **targets,
    float **meta,
    int unknown_id,
    int docs_count)
{
    int individual_size = (int)individual[0];
    int entity_id = (int)individual[2];
    float perfomance_doc[docs_count];
    float individualRep[individual_size];
    for (int i = 0; i < individual_size; i++)
    {
        individualRep[i] = individual[i + 3];
    }
    for (int index_doc = 0; index_doc < docs_count; index_doc++)
    {
        perfomance_doc[index_doc] = perfomance_by_doc(
            individualRep,
            docs[index_doc],
            targets[index_doc],
            (int)meta[index_doc][0],
            individual_size,
            unknown_id,
            entity_id);
    }

    float sum = 0.0;
    int count = 0;
    for (int i = 0; i < docs_count; i++)
    {
        if (perfomance_doc[i] >= 0)
        {
            sum += perfomance_doc[i];
            count++;
        }
    }

    if (count == 0)
    {
        return 0.0;
    }
    return sum / count;
}

float *fitness(
    float **population,
    float ***docs,
    int **targets,
    float **meta,
    int unknown_id,
    int docs_count,
    int population_size,
    int n_threads)
{

    float *fitness_array = new float[population_size];
#pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < population_size; i++)
    {
        fitness_array[i] = fitness_by_individual(
            population[i],
            docs,
            targets,
            meta,
            unknown_id,
            docs_count);
    }
    return fitness_array;
}
