//%%writefile evolutionary_operations.cu
#include "evolutionary_operations.cuh"
#include "omp.h"
#include <stdio.h>
#include <math.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// Function to generate random numbers in given range
__host__ int random_num(int start, int end)
{
    int range = (end - start) + 1;
    int random_int = range > 0 ? start + (rand() % range) : start;
    return random_int;
}

__device__ float perfomance_by_doc(
    float *population,
    float *docs,
    int *targets,
    int doc_size,
    int individual_size,
    int unknown_id,
    int entity_id,
    const int indexIndividual,
    const int populationDim,
    const int index_doc,
    const int docsDim,
    const int docLen)
{

    if (individual_size > doc_size + 2)
    {
        return 0.0;
    }

    // Print the individual, document and tartget if individual_size > 1
    // if (individual_size > 1)
    // {
    //     printf("Individual: %d %d %d %d %d %d %d %d %d %d\n", (int)population[indexIndividual * populationDim], (int)population[indexIndividual * populationDim + 1], (int)population[indexIndividual * populationDim + 2], (int)population[indexIndividual * populationDim + 3], (int)population[indexIndividual * populationDim + 4], (int)population[indexIndividual * populationDim + 5], (int)population[indexIndividual * populationDim + 6], (int)population[indexIndividual * populationDim + 7], (int)population[indexIndividual * populationDim + 8], (int)population[indexIndividual * populationDim + 9]);
    // }

    int unionDoc = 0;
    int intercepDoc = 0;
    int retriveDoc = 0;

    // Slice doc
    int startDoc = index_doc * docLen;
    int endDoc = startDoc + (doc_size - individual_size + 2);
    for (int indexTokenDoc = startDoc; indexTokenDoc < endDoc; indexTokenDoc++)
    {
        // Slice tokens
        bool isMatch = true;
        // Individual features start at 3
        int startIndividual = indexIndividual * populationDim;
        for (int indexTokenIndividual = 0; indexTokenIndividual < individual_size; indexTokenIndividual++)
        {
            bool anyMatch = false;
            for (int indexFeature = 0; indexFeature < docsDim; indexFeature++)
            {
                int token = docs[indexTokenDoc * docsDim + indexTokenIndividual * docsDim + indexFeature];
                int IndividualFeature = population[startIndividual + indexTokenIndividual + 3];
                anyMatch = anyMatch || token == abs(IndividualFeature) || IndividualFeature == unknown_id;
            }
            isMatch = isMatch && anyMatch;
        }
        //  Evaluate match with target
        if (isMatch)
        {
            for (int indexTokenIndividual = 0; indexTokenIndividual < individual_size; indexTokenIndividual++)
            {
                bool targetToken = targets[indexTokenDoc + indexTokenIndividual] == entity_id;
                bool retriveToken = population[startIndividual + indexTokenIndividual + 3] > 0;
                // printf("\nretriveToken: %d, index: %d - ", retriveToken, startIndividual + indexTokenIndividual + 3);
                unionDoc += int(targetToken || retriveToken);
                intercepDoc += int(targetToken && retriveToken);
                retriveDoc += int(retriveToken);
            }
        }
    }

    // printf("unionDoc: %d, intercepDoc: %d, retriveDoc: %d", unionDoc, intercepDoc, retriveDoc);

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

__device__ float fitness_by_individual(
    float *population,
    float *docs,
    int *targets,
    float *meta,
    int unknown_id,
    const int docs_count,
    const int indexIndividual,
    const int populationDim,
    const int docsDim,
    const int docLen,
    const int metaDim)
{
    const int individual_size = (int)population[indexIndividual * populationDim];
    int entity_id = (int)population[indexIndividual * populationDim + 2];
    // float perfomance_doc[docs_count];
    float perfomance_docs = 0.0;
    int count = 0;
    for (int index_doc = 0; index_doc < docs_count; index_doc++)
    {
        int tempPerfomance = perfomance_by_doc(
            population,
            docs,
            targets,
            (int)meta[index_doc * metaDim],
            individual_size,
            unknown_id,
            entity_id,
            indexIndividual,
            populationDim,
            index_doc,
            docsDim,
            docLen);
        if (tempPerfomance >= 0)
        {
            perfomance_docs += tempPerfomance;
            count++;
        }
    }

    if (count == 0)
    {
        return 0.0;
    }
    return perfomance_docs / count;
}

__device__ void swap_fitness(
    float *population,
    float *population_fitness,
    int index,
    float *population2,
    float *population2_fitness,
    int index2,
    const int populationDim)
{
    float temp_swap;
    float temp_swap_fitness;
    for (int i = 0; i < populationDim; i++)
    {
        temp_swap = population[index * populationDim + i];
        population[index * populationDim + i] = population2[index2 * populationDim + i];
        population2[index2 * populationDim + i] = temp_swap;
    }
    temp_swap_fitness = population_fitness[index];
    population_fitness[index] = population2_fitness[index2];
    population2_fitness[index2] = temp_swap_fitness;
}

__device__ void copy_individual(
    float *population,
    float *population2,
    int index,
    int index2,
    const int populationDim)
{
    for (int i = 0; i < populationDim; i++)
    {
        population2[index2 * populationDim + i] = population[index * populationDim + i];
    }
}

/**
 * @brief
 *
 * @param population
 * @param population_offspring
 * @param indexIndividual
 * @param populationDim
 * @param state
 * @return __device__
 *
 * Operation:
 *  0 - Add. Create 2 copies of the individual, one without posive gen and other without negative gen new feature
 *  1 - Remove
 *  2 - Replace Create 2 copies of the individual, one without posive gen and other without negative gen selected feature
 */
/**
 * @brief
 *
 * @param population
 * @param population_offspring
 * @param indexIndividual
 * @param populationDim
 * @param state
 * @return __device__
 *
 * Operation:
 *  0 - Add. Create 2 copies of the individual, one without posive gen and other without negative gen new feature
 *  1 - Remove
 *  2 - Replace Create 2 copies of the individual, one without posive gen and other without negative gen selected feature
 */
__device__ void mutate(
    float *population,
    float *population_offspring,
    const int indexIndividual,
    const int populationDim,
    curandState_t *state)
{

    int size_individual = (int)population[indexIndividual * populationDim];
    int operation = curand_uniform(state) * 3;
    int index_feature = 0;
    int indexIndividualOffspring1 = indexIndividual * 2;
    int indexIndividualOffspring2 = indexIndividual * 2 + 1;

    copy_individual(
        population,
        population_offspring,
        indexIndividual,
        indexIndividualOffspring1,
        populationDim);

    if (size_individual <= 1 && operation == 1)
    {
        operation = 0;
        index_feature = size_individual + 3;
    }
    else if (size_individual == 1 && operation == 2)
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
    // Select random index feature
    if (index_feature == 0)
    {
        index_feature = curand_uniform(state) * size_individual + 3;
    }

    if (operation == 0)
    {
        copy_individual(
            population,
            population_offspring,
            indexIndividual,
            indexIndividualOffspring2,
            populationDim);
        population_offspring[indexIndividualOffspring1 * populationDim] = size_individual;
        population_offspring[indexIndividualOffspring2 * populationDim] = size_individual;
        // Hold the data of the individual up to the index_feature
        float next_val;
        for (int index = 3 + size_individual - 1; index > index_feature; index--)
        {
            next_val = population[indexIndividual * populationDim + index - 1];
            population_offspring[indexIndividualOffspring1 * populationDim + index] = next_val;
            population_offspring[indexIndividualOffspring2 * populationDim + index] = next_val;
        }
    }
    else if (operation == 1)
    {
        size_individual -= 1;
        population_offspring[indexIndividualOffspring1 * populationDim] = size_individual;

        float next_val;
        for (int index = index_feature; index < 3 + size_individual; index++)
        {
            next_val = population[indexIndividual * populationDim + index + 1];
            population_offspring[indexIndividualOffspring1 * populationDim + index] = next_val;
        }
        for (int index = 3 + size_individual; index < populationDim; index++)
        {
            population_offspring[indexIndividualOffspring1 * populationDim + index] = 0;
        }
    }
    else if (operation == 2)
    {
        copy_individual(
            population,
            population_offspring,
            indexIndividual,
            indexIndividualOffspring2,
            populationDim);
    }

    // Create new gen
    if (operation == 0 || operation == 2)
    {
        float gen = (int)(curand_uniform(state) * 10) + 1;
        population_offspring[indexIndividualOffspring1 * populationDim + index_feature] = gen;
        population_offspring[indexIndividualOffspring2 * populationDim + index_feature] = -gen;
    }
    // int size1 = (int)population_offspring[indexIndividualOffspring1 * populationDim];
    // if (operation == 0 || operation == 2)
    // {
    //     int size2 = (int)population_offspring[indexIndividualOffspring2 * populationDim];
    //     if (size1 > 7 || size2 > 7 || size1 < 1 || size2 < 1)
    //     {
    //         printf("operation: %d, size1: %d index_feature %d, %d %d %d %d %d %d %d", operation, size1, index_feature, (int)population_offspring[indexIndividualOffspring1 * populationDim + 3], (int)population_offspring[indexIndividualOffspring1 * populationDim + 4], (int)population_offspring[indexIndividualOffspring1 * populationDim + 5], (int)population_offspring[indexIndividualOffspring1 * populationDim + 6], (int)population_offspring[indexIndividualOffspring1 * populationDim + 7], (int)population_offspring[indexIndividualOffspring1 * populationDim + 8], (int)population_offspring[indexIndividualOffspring1 * populationDim + 9]);
    //         printf("operation: %d, size2: %d, %d %d %d %d %d %d %d", operation, size2, (int)population_offspring[indexIndividualOffspring2 * populationDim + 3], (int)population_offspring[indexIndividualOffspring2 * populationDim + 4], (int)population_offspring[indexIndividualOffspring2 * populationDim + 5], (int)population_offspring[indexIndividualOffspring2 * populationDim + 6], (int)population_offspring[indexIndividualOffspring2 * populationDim + 7], (int)population_offspring[indexIndividualOffspring2 * populationDim + 8], (int)population_offspring[indexIndividualOffspring2 * populationDim + 9]);
    //     }
    //     else if (size1 > 7 || size1 < 1)
    //     {
    //         printf("operation: %d, size1: %d, index_feature %d, %d %d %d %d %d %d %d", operation, size1, index_feature, (int)population_offspring[indexIndividualOffspring1 * populationDim + 3], (int)population_offspring[indexIndividualOffspring1 * populationDim + 4], (int)population_offspring[indexIndividualOffspring1 * populationDim + 5], (int)population_offspring[indexIndividualOffspring1 * populationDim + 6], (int)population_offspring[indexIndividualOffspring1 * populationDim + 7], (int)population_offspring[indexIndividualOffspring1 * populationDim + 8], (int)population_offspring[indexIndividualOffspring1 * populationDim + 9]);
    //     }
    // }
}

__device__ float cosine_similarity(
    float *vector1,
    float *vector2,
    int vectorDim,
    int startIndex1,
    int startIndex2)
{
    float sum = 0;
    float sum1 = 0;
    float sum2 = 0;
    for (int i = 0; i < vectorDim; i++)
    {
        sum += vector1[startIndex1 + i] * vector2[startIndex2 + i];
        sum1 += vector1[startIndex1 + i] * vector1[startIndex1 + i];
        sum2 += vector2[startIndex2 + i] * vector2[startIndex2 + i];
    }

    return sum / (sqrt(sum1) * sqrt(sum2));
}

__device__ float individual_shared_fitness(
    int index,
    float *population,
    float *population2,
    int populationDim,
    int populationSize,
    int population2Size,
    const int numIslands)
{

    float num_members = 0;
    int index_start_ind = 2;
    int vectorDim = populationDim - index_start_ind;
    int indexIndividual = index * populationDim + index_start_ind;
    // Displacement for population
    int populationIslandSize = populationSize / numIslands;
    int displacement = index / populationIslandSize;

    for (int index2 = displacement; index2 < populationIslandSize + displacement; index2++)
    {
        float simil = cosine_similarity(
            population,
            population,
            vectorDim,
            indexIndividual,
            index2 * populationDim + index_start_ind);
        if (simil > 0.5)
        {
            num_members += simil;
        }
    }

    // Displacement for population2
    int population2IslandSize = population2Size / numIslands;
    displacement = index / population2IslandSize;
    for (int index2 = displacement; index2 < population2IslandSize + displacement; index2++)
    {
        float simil = cosine_similarity(
            population,
            population2,
            vectorDim,
            indexIndividual,
            index2 * populationDim + index_start_ind);
        if (simil > 0.5)
        {
            num_members += simil;
        }
    }

    return num_members;
}

__device__ void init_end_iteration(
    const int index,
    const int numElements,
    const int totalThreads,
    int &start,
    int &end)
{
    if (numElements <= totalThreads)
    {
        if (index < numElements)
        {
            start = index;
            end = index + 1;
        }
        else
        {
            start = 0;
            end = 0;
        }
    }
    else
    {
        float numElementsPerThread = (float)numElements / totalThreads;
        start = (int)(index * numElementsPerThread);
        end = (int)((index + 1) * numElementsPerThread);
        if (end > numElements)
        {
            end = numElements;
        }
    }
}

__global__ void kernel_train(
    float *population,
    float *population_offspring,
    float *docs,
    int *targets,
    float *meta,
    int unknown_id,
    int docs_count,
    int population_size,
    int populationDim,
    int docsDim,
    int docLen,
    int metaDim,
    int n_threads,
    float *fitness_array,
    float *fitness_spring,
    curandState_t *states,
    const int numIslands)
{

    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int initIteration, endIteration;

    init_end_iteration(
        globalIndex,
        population_size * 3,
        n_threads,
        initIteration,
        endIteration);

    for (int index = initIteration; index < endIteration; index++)
    {
        if (index < population_size)
        {
            mutate(
                population,
                population_offspring,
                index,
                populationDim,
                &states[index]);
        }
        __syncthreads();

        // Fitness parent population
        if (index < population_size)
        {
            fitness_array[index] = fitness_by_individual(
                population,
                docs,
                targets,
                meta,
                unknown_id,
                docs_count,
                index,
                populationDim,
                docsDim,
                docLen,
                metaDim);
        }
        // Fitness offspring population
        if (index >= population_size && index < population_size * 3)
        {
            fitness_spring[index - population_size] = fitness_by_individual(
                population_offspring,
                docs,
                targets,
                meta,
                unknown_id,
                docs_count,
                index - population_size,
                populationDim,
                docsDim,
                docLen,
                metaDim);
        }
        __syncthreads();
        // Selection
        // Fitness shared using cosine similarity
        if (index < population_size)
        {
            int num_members = individual_shared_fitness(
                index,
                population,
                population_offspring,
                populationDim,
                population_size,
                population_size * 2,
                numIslands);
            fitness_array[index] = fitness_array[index] / num_members;
        }

        if (index >= population_size && index < population_size * 3)
        {
            int num_members = individual_shared_fitness(
                index - population_size,
                population_offspring,
                population,
                populationDim,
                population_size * 2,
                population_size,
                numIslands);
            fitness_spring[index - population_size] = fitness_spring[index - population_size] / num_members;
        }
    }
}

__global__ void kernel_population_sort(
    float *population,
    float *population_offspring,
    float *fitness_array,
    float *fitness_spring,
    int population_size,
    int populationDim,
    int n_threads,
    int numIslands)
{
    int localIndex = threadIdx.x;
    int initIteration, endIteration;

    init_end_iteration(
        localIndex,
        population_size,
        blockDim.x,
        initIteration,
        endIteration);

    for (int index = initIteration; index < endIteration; index++)
    {
        int index1 = index + population_size * blockIdx.x;
        if (index + 1 < population_size && index1 % 2 == 0)
        {
            // Horizontal swap
            if (fitness_array[index1] < fitness_array[index1 + 1])
            {
                swap_fitness(
                    population,
                    fitness_array,
                    index1,
                    population,
                    fitness_array,
                    index1 + 1,
                    populationDim);
            }
            if (fitness_spring[index1] > fitness_spring[index1 + 1])
            {
                swap_fitness(
                    population_offspring,
                    fitness_spring,
                    index1,
                    population_offspring,
                    fitness_spring,
                    index1 + 1,
                    populationDim);
            }
            if (fitness_spring[index1 + population_size] > fitness_spring[index1 + population_size + 1])
            {
                swap_fitness(
                    population_offspring,
                    fitness_spring,
                    index1 + population_size,
                    population_offspring,
                    fitness_spring,
                    index1 + population_size + 1,
                    populationDim);
            }

            // Vertical swap
            if (fitness_array[index1] < fitness_spring[index1])
            {
                swap_fitness(
                    population,
                    fitness_array,
                    index1,
                    population_offspring,
                    fitness_spring,
                    index1,
                    populationDim);
            }

            if (fitness_array[index1] < fitness_spring[index1 + population_size])
            {
                swap_fitness(
                    population,
                    fitness_array,
                    index1,
                    population_offspring,
                    fitness_spring,
                    index1 + population_size,
                    populationDim);
            }
        }
        __syncthreads();
        if (index + 1 < population_size && index1 % 2 == 1)
        {
            // Horizontal swap
            if (fitness_array[index1] < fitness_array[index1 + 1])
            {
                swap_fitness(
                    population,
                    fitness_array,
                    index1,
                    population,
                    fitness_array,
                    index1 + 1,
                    populationDim);
            }
            if (fitness_spring[index1] > fitness_spring[index1 + 1])
            {
                swap_fitness(
                    population_offspring,
                    fitness_spring,
                    index1,
                    population_offspring,
                    fitness_spring,
                    index1 + 1,
                    populationDim);
            }
            if (fitness_spring[index1 + population_size] > fitness_spring[index1 + population_size + 1])
            {
                swap_fitness(
                    population_offspring,
                    fitness_spring,
                    index1 + population_size,
                    population_offspring,
                    fitness_spring,
                    index1 + population_size + 1,
                    populationDim);
            }

            // Vertical swap
            if (fitness_array[index1] < fitness_spring[index1])
            {
                swap_fitness(
                    population,
                    fitness_array,
                    index1,
                    population_offspring,
                    fitness_spring,
                    index1,
                    populationDim);
            }

            if (fitness_array[index1] < fitness_spring[index1 + population_size])
            {
                swap_fitness(
                    population,
                    fitness_array,
                    index1,
                    population_offspring,
                    fitness_spring,
                    index1 + population_size,
                    populationDim);
            }
        }
        __syncthreads();
    }
}

/* this GPU kernel function is used to initialize the random states */
__global__ void init_random_states(unsigned int seed, unsigned int n, unsigned int totalThreads, curandState_t *states)
{
    int initIteration, endIteration;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n)
    {
        if (n < totalThreads)
        {
            initIteration = index;
            endIteration = index + 1;
        }
        else
        {
            float increment = (float)n / totalThreads;
            initIteration = increment * index;
            endIteration = increment * (index + 1);
        }
        for (int i = initIteration; i < endIteration; i++)
        {
            /* we have to initialize the state */
            curand_init(seed, i, 0, &states[i]);
        }
    }
}

void train(
    float **population,
    float ***docs,
    int **targets,
    float **meta,
    int unknown_id,
    int docs_count,
    int population_size,
    int populationDim,
    int docsDim,
    int docLen,
    int metaDim,
    int n_threads,
    int blocksPerGrid,
    float *h_fitness_array,
    curandState_t *states)
{
    float *d_population;
    float *d_population_offspring;
    float *d_docs;
    int *d_targets;
    float *d_meta;
    float *d_fitness_array;
    float *d_fitness_spring;
    float *h_population;
    float *h_docs;
    int *h_targets;
    float *h_meta;

    // Allocate memory
    cudaMalloc((void **)&d_population, population_size * populationDim * sizeof(float));
    cudaMalloc((void **)&d_population_offspring, population_size * populationDim * sizeof(float) * 2);
    cudaMalloc((void **)&d_docs, docs_count * docsDim * docLen * sizeof(float));
    cudaMalloc((void **)&d_targets, docs_count * docLen * sizeof(int));
    cudaMalloc((void **)&d_meta, docs_count * metaDim * sizeof(float));
    cudaMalloc((void **)&d_fitness_array, population_size * sizeof(float));
    cudaMalloc((void **)&d_fitness_spring, population_size * sizeof(float) * 2);

    // Copy data to device
    h_population = new float[population_size * populationDim];
    h_docs = new float[docs_count * docsDim * docLen];
    h_targets = new int[docs_count * docLen];
    h_meta = new float[docs_count * metaDim];
    // Copy population host
    for (int i = 0; i < population_size; i++)
    {
        for (int j = 0; j < populationDim; j++)
        {
            h_population[i * populationDim + j] = population[i][j];
        }
    }
    // Copy docs host
    for (int i = 0; i < docs_count; i++)
    {
        for (int j = 0; j < docLen; j++)
        {
            for (int k = 0; k < docsDim; k++)
            {
                h_docs[i * docsDim * docLen + j * docsDim + k] = docs[i][j][k];
            }
        }
    }
    // Copy targets host
    for (int i = 0; i < docs_count; i++)
    {
        for (int j = 0; j < docLen; j++)
        {
            h_targets[i * docLen + j] = targets[i][j];
        }
    }
    // Copy meta host
    for (int i = 0; i < docs_count; i++)
    {
        for (int j = 0; j < metaDim; j++)
        {
            h_meta[i * metaDim + j] = meta[i][j];
        }
    }
    cudaMemcpy(d_population, h_population, population_size * populationDim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_docs, h_docs, docs_count * docsDim * docLen * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, docs_count * docLen * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_meta, h_meta, docs_count * metaDim * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel
    // int threadsPerBlock = n_threads;
    // int blocksPerGrid = (population_size + threadsPerBlock - 1) / threadsPerBlock;

    // kernel_train<<<blocksPerGrid, threadsPerBlock>>>(
    int threadsPerBlock;
    int numIsland = 1;

    for (int k = 0; k < 1; k++)
    {
        threadsPerBlock = (int)ceil((float)n_threads / (float)blocksPerGrid);
        kernel_train<<<blocksPerGrid, threadsPerBlock>>>(
            d_population,
            d_population_offspring,
            d_docs,
            d_targets,
            d_meta,
            unknown_id,
            docs_count,
            population_size,
            populationDim,
            docsDim,
            docLen,
            metaDim,
            n_threads,
            d_fitness_array,
            d_fitness_spring,
            states,
            numIsland);
        cudaDeviceSynchronize();
        threadsPerBlock = (int)ceil((float)n_threads / (float)numIsland);
        kernel_population_sort<<<numIsland, threadsPerBlock>>>(
            d_population,
            d_population_offspring,
            d_fitness_array,
            d_fitness_spring,
            population_size / numIsland,
            populationDim,
            n_threads,
            numIsland);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(h_fitness_array, d_fitness_array, population_size * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_population, d_population, population_size * populationDim * sizeof(float), cudaMemcpyDeviceToHost));
    // Copy data from h_population to population
    for (int i = 0; i < population_size; i++)
    {
        for (int j = 0; j < populationDim; j++)
        {
            population[i][j] = h_population[i * populationDim + j];
        }
    }

    checkCudaErrors(cudaFree(d_population));
    checkCudaErrors(cudaFree(d_population_offspring));
    checkCudaErrors(cudaFree(d_docs));
    checkCudaErrors(cudaFree(d_targets));
    checkCudaErrors(cudaFree(d_meta));
    checkCudaErrors(cudaFree(d_fitness_array));
    checkCudaErrors(cudaFree(d_fitness_spring));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

int main(int argc, char *argv[])
{
    int population_size = 100;
    int populationDim = 10;
    int docsDim = 3;
    int docLen = 100;
    int metaDim = 3;
    int docs_count = 2;
    int totalThreads = 100;
    int unknown_id = 7;
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

    int dev = 0;

    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    curandState_t *states;
    checkCudaErrors(cudaMalloc((void **)&states, population_size * sizeof(curandState_t)));
    int blocksPerGrid = deviceProp.multiProcessorCount;
    int threadsPerBlock = (int)ceil((float)totalThreads / (float)blocksPerGrid);
    init_random_states<<<blocksPerGrid, threadsPerBlock>>>(42, population_size, totalThreads, states);

    train(
        population,
        docs,
        targets,
        meta,
        unknown_id,
        docs_count,
        population_size,
        populationDim,
        docsDim,
        docLen,
        metaDim,
        totalThreads,
        blocksPerGrid,
        fitness_array,
        states);

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

    checkCudaErrors(cudaFree(states));

    return 0;
}

__global__ void test()
{
    printf("Hi Cuda World");
}
