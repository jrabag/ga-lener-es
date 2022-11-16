#include "evolutionary_operations.h"
#include "omp.h"
#include <stdio.h>
#include <math.h>

// Function to generate random numbers in given range
int random_num(int start, int end)
{
    int range = (end - start) + 1;
    int random_int = range > 0 ? start + (rand() % range) : start;
    return random_int;
}

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

    return (float)intercepDoc / retriveDoc * log2(retriveDoc + 0.0001);
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

void copy_individual(
    float *individual1,
    float *population2)
{
    int individual_size = (int)individual1[0];
    for (int i = 0; i < individual_size + 3; i++)
    {
        population2[i] = individual1[i];
    }
}

void swap_fitness(
    float **population,
    float *fitness,
    int index1,
    float **population2,
    float *fitness2,
    int index2)
{
    float *temp = population[index1];
    population[index1] = population2[index2];
    population2[index2] = temp;
    float temp2 = fitness[index1];
    fitness[index1] = fitness2[index2];
    fitness2[index2] = temp2;
}

/**
 * @brief
 *
 * @param individual
 * @param individual1
 * @param individual2
 * @param populationDim
 *
 * Operation:
 *  0 - Add. Create 2 copies of the individual, one without posive gen and other without negative gen new feature
 *  1 - Remove
 *  2 - Replace Create 2 copies of the individual, one without posive gen and other without negative gen selected feature
 */
void mutate(
    float **population,
    float **population2,
    const int index,
    const int populationDim)
{
    int indexIndividualOffspring1 = index * 2;
    int indexIndividualOffspring2 = index * 2 + 1;
    float *individual = population[index];
    float *individual1 = population[index];
    float *individual2 = population[index];

    int size_individual = (int)individual[0];
    int operation = rand() % 3;
    int index_feature = 0;
    if (size_individual == 1 && operation == 1)
    {
        operation = 0;
        index_feature = size_individual + 3;
    }
    if (size_individual == 1 && operation == 2)
    {
        index_feature = 3;
    }
    else if (operation == 0 && size_individual + 3 >= populationDim)
    {
        operation = 1;
        index_feature = populationDim - 1;
    }

    if (operation == 0)
    {
        size_individual += 1;
    }

    if (index_feature == 0)
    {
        index_feature = (rand() % size_individual) + 3;
    }

    if (operation == 0)
    {
        individual1[0] = size_individual;

        for (int index = 3 + size_individual - 1; index >= index_feature; index--)
        {
            float next_val = individual[index - 1];
            individual1[index] = next_val;
        }
    }
    else if (operation == 1)
    {
        size_individual -= 1;
        individual1[0] = size_individual;
        individual2[0] = size_individual;
        for (int index = index_feature; index < 3 + size_individual; index++)
        {
            float next_val = individual[index + 1];
            individual1[index] = next_val;
            individual2[index] = next_val;
        }

        individual1[3 + size_individual] = 0;
        individual2[3 + size_individual] = 0;
    }

    if (operation == 0 || operation == 2)
    {
        int new_size = (int)individual1[0] + 3;
        float gen = rand() % 10; // 2784;
        individual1[index_feature] = gen;
        individual2[index_feature] = -gen;
        population2[indexIndividualOffspring1] = individual1;
        population2[indexIndividualOffspring2] = individual2;
    }
    else if (operation == 1)
    {
        population2[indexIndividualOffspring1] = individual1;
    }
}

float cosine_similarity(
    float *vector1,
    float *vector2,
    int vectorDim,
    int startIndex)
{
    float sum = 0;
    float sum1 = 0;
    float sum2 = 0;
    for (int i = 0; i < vectorDim; i++)
    {
        sum += vector1[startIndex + i] * vector2[startIndex + i];
        sum1 += vector1[startIndex + i] * vector1[startIndex + i];
        sum2 += vector2[startIndex + i] * vector2[startIndex + i];
    }

    return sum / (sqrt(sum1) * sqrt(sum2));
}

float individual_shared_fitness(
    int index,
    float **population,
    float **population2,
    int populationDim,
    int populationSize,
    int population2Size,
    float &fitness)
{

    float num_members = 0;
    int index_start_ind = 2;
    int vectorDim = populationDim - index_start_ind;

    for (int index2 = 0; index2 < populationSize; index2++)
    {
        float simil = cosine_similarity(
            population[index],
            population[index2],
            vectorDim,
            index_start_ind);
        if (simil > 0.5)
        {
            num_members += simil;
        }
    }

    for (int index2 = 0; index2 < population2Size; index2++)
    {
        float simil = cosine_similarity(
            population[index],
            population2[index2],
            vectorDim,
            index_start_ind);
        if (simil > 0.5)
        {
            num_members += simil;
        }
    }

    fitness /= num_members;

    return num_members;
}

void select_population(
    float **population,
    float *fitness,
    float **population2,
    float *fitness2,
    int n_population)
{
    for (int index = 0; index < n_population; index++)
    {
        // Horizontal swap
        if (index + 1 < n_population)
        {
            if (fitness[index] < fitness[index + 1])
            {
                swap_fitness(
                    population,
                    fitness,
                    index,
                    population,
                    fitness,
                    index + 1);
            }
            if (fitness2[index] > fitness2[index + 1])
            {
                swap_fitness(
                    population2,
                    fitness2,
                    index,
                    population2,
                    fitness2,
                    index + 1);
            }
            if (fitness2[index + n_population] > fitness2[index + n_population + 1])
            {
                swap_fitness(
                    population2,
                    fitness2,
                    index + n_population,
                    population2,
                    fitness2,
                    index + n_population + 1);
            }
        }
        // Vertical swap
        if (fitness[index] < fitness2[index])
        {
            swap_fitness(
                population,
                fitness,
                index,
                population2,
                fitness2,
                index);
        }
        if (fitness[index] < fitness2[index + n_population])
        {
            swap_fitness(
                population,
                fitness,
                index,
                population2,
                fitness2,
                index + n_population);
        }
    }
}

void train_step(
    float **population,
    float *fitness,
    float **population2,
    float *fitness2,
    int n_population,
    int populationDim,
    float ***docs,
    int **targets,
    float **meta,
    int unknown_id,
    int n_docs)
{
    for (int index = 0; index < n_population; index++)
    {
        mutate(population, population2, index, populationDim);
    }

    // Calculate fitness of offspring
    for (int index = 0; index < n_population * 2; index++)
    {
        fitness2[index] = fitness_by_individual(
            population2[index],
            docs,
            targets,
            meta,
            unknown_id,
            n_docs);
    }

    for (int index = 0; index < n_population; index++)
    {
        fitness[index] = fitness_by_individual(
            population[index],
            docs,
            targets,
            meta,
            unknown_id,
            n_docs);
    }

    // Shared fitness
    for (int index = 0; index < n_population * 2; index++)
    {
        individual_shared_fitness(
            index,
            population,
            population2,
            populationDim,
            n_population,
            n_population * 2,
            fitness2[index]);
    }

    for (int index = 0; index < n_population; index++)
    {
        individual_shared_fitness(
            index,
            population,
            population2,
            populationDim,
            n_population,
            n_population * 2,
            fitness[index]);
    }

    select_population(
        population,
        fitness,
        population2,
        fitness2,
        n_population);
}

// C code
void train(
    float **population,
    float *fitness,
    int n_population,
    int populationDim,
    float ***docs,
    int **targets,
    float **meta,
    int unknown_id,
    int n_docs,
    int max_iter,
    int tol,
    int num_islands,
    int num_threads)
{
    int n_not_improve = 0;
    float mean_best_fitness = 0;
    int i = 0;

    float **population2 = (float **)malloc(sizeof(float *) * n_population * 2);
    float *fitness2 = (float *)malloc(sizeof(float) * n_population * 2);

    for (i = 0; i < max_iter; i++)
    {
        int step_range = n_population / num_islands;
#pragma omp parallel for num_threads(num_threads)
        for (int index_island = 0; index_island < num_islands; index_island++)
        {
            int start = index_island * step_range;
            int end = start + step_range;
            if (end > n_population)
            {
                end = n_population;
            }
            float **population_island = population + start;
            float *fitness_island = fitness + start;
            float **population2_island = population2 + start * 2;
            float *fitness2_island = fitness2 + start * 2;

            train_step(
                population_island,
                fitness_island,
                population2_island,
                fitness2_island,
                end - start,
                populationDim,
                docs,
                targets,
                meta,
                unknown_id,
                n_docs);
        }

        // Migration
        if (num_islands > 1 && i % 10 == 0)
        {
            for (int index_island = 0; index_island < num_islands; index_island++)
            {
                int index_migration = rand() % step_range + index_island * step_range;
                int index_accept = rand() % step_range + (index_island + 1) * step_range;
                if (index_accept >= n_population)
                {
                    index_accept = index_accept - n_population;
                }
                if (fitness[index_migration] > fitness[index_accept])
                {
                    for (int j = 0; j < populationDim; j++)
                    {
                        population[index_accept][j] = population[index_migration][j];
                    }
                    fitness[index_accept] = fitness[index_migration];
                }
            }
        }

        float mean_fitness = 0;
        for (int j = 0; j < n_population; j++)
        {
            mean_fitness += fitness[j];
        }
        mean_fitness /= n_population;
        float max_curr_fitness = 0;
        for (int j = 0; j < n_population; j++)
        {
            if (fitness[j] > max_curr_fitness)
            {
                max_curr_fitness = fitness[j];
            }
        }
    }

    free(population2);
    free(fitness2);
}

int main(int argc, char *argv[])
{
    int population_size = 100;
    int populationDim = 10;
    int docsDim = 3;
    int docLen = 100;
    int metaDim = 3;
    int docs_count = 2;
    int totalThreads = 8;
    int unknown_id = 7;
    int max_iter = 100;
    int tol = 10;
    int num_islands = 2;
    srand(42);

    float **population = new float *[population_size];
    for (int i = 0; i < population_size; i++)
    {
        population[i] = new float[populationDim];
        population[i][0] = 1;
        population[i][1] = 0;
        population[i][2] = 1;
        population[i][3] = random_num(1, 10);
        population[i][4] = 0;
        population[i][5] = 0;
        population[i][6] = 0;
        population[i][7] = 0;
        population[i][8] = 0;
        population[i][9] = 0;
    }

    float ***docs = new float **[docs_count];
    for (int i = 0; i < docs_count; i++)
    {
        docs[i] = new float *[docLen];
        for (int j = 0; j < docLen; j++)
        {
            docs[i][j] = new float[docsDim];
            docs[i][j][0] = random_num(1, 2);
            docs[i][j][1] = random_num(3, 5);
            docs[i][j][2] = random_num(5, 10);
        }
    }

    int **targets = new int *[docs_count];
    for (int i = 0; i < docs_count; i++)
    {
        targets[i] = new int[docLen];
        for (int j = 0; j < docLen; j++)
        {
            // targets[i][j] = random_num(0, 1);
            targets[i][j] = 1;
        }
    }

    float **meta = new float *[docs_count];
    for (int i = 0; i < docs_count; i++)
    {
        meta[i] = new float[metaDim];
        meta[i][0] = 93;
        meta[i][1] = 0;
        meta[i][2] = 0;
    }

    float *fitness_array = new float[population_size];
    for (int i = 0; i < population_size; i++)
    {
        fitness_array[i] = 0;
    }

    train(
        population,
        fitness_array,
        population_size,
        populationDim,
        docs,
        targets,
        meta,
        unknown_id,
        docs_count,
        max_iter,
        tol,
        num_islands,
        totalThreads);
    // print fitness array
    printf("\n");
    for (int i = 0; i < population_size; i++)
    {
        printf("%f  ", fitness_array[i]);
    }
    // print population
    for (int i = 0; i < population_size; i++)
    {
        for (int j = 0; j < populationDim; j++)
        {
            printf("%d  ", (int)population[i][j]);
        }
        printf("\n");
    }

    return 0;
}