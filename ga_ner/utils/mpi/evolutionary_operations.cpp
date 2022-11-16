// mpicxx -o evo evolutionary_operations.cpp
// mpirun -np 2 evo

#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#define NUM_TAGS 4
#define POS_SIZE 19
#define DEP_SIZE 62
#define WORD_SIZE 2703
#define VOCAB_SIZE 2784
#define UNK_ID 2781

// Function to generate random numbers in given range
int random_num(int start, int end)
{
    int range = (end - start) + 1;
    int random_int = range > 0 ? start + (rand() % range) : start;
    return random_int;
}

float perfomance_by_doc(
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
    const int docLen,
    int entityLabel)
{

    // Validate if target is has entity label
    bool hasEntity = false;
    for (int i = 0; i < docLen; i++)
    {
        if (targets[index_doc + i] == entityLabel)
        {
            hasEntity = true;
            break;
        }
    }

    if (!hasEntity)
    {
        return -1;
    }

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

float fitness_by_individual(
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
    const int metaDim,
    const int entityLabel)
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
            docLen,
            entityLabel);
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

void swap_fitness(
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

void copy_individual(
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
void mutate(
    float *population,
    float *population_offspring,
    const int indexIndividual,
    const int populationDim)
{

    int size_individual = (int)population[indexIndividual * populationDim];
    int operation = rand() % 3;
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
        index_feature = rand() % size_individual + 3;
    }

    if (operation == 0)
    {
        // Add
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
        float gen;
        int featureType = rand() % 3;
        if (featureType == 0)
        {
            gen = random_num(2, POS_SIZE);
        }
        else if (featureType == 1)
        {
            gen = random_num(POS_SIZE + 2, POS_SIZE + DEP_SIZE);
        }
        else
        {
            gen = random_num(POS_SIZE + DEP_SIZE, VOCAB_SIZE);
        }

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

float cosine_similarity(
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

float individual_shared_fitness(
    int index,
    float *population,
    float *population2,
    int populationDim,
    int populationSize,
    int population2Size,
    const int islandNumber)
{

    float num_members = 0;
    int index_start_ind = 2;
    int vectorDim = populationDim - index_start_ind;
    int indexIndividual = index * populationDim + index_start_ind;
    // Displacement for population
    int displacement = islandNumber * populationSize;

    for (int index2 = displacement; index2 < populationSize + displacement; index2++)
    {
        float simil = cosine_similarity(
            population,
            population,
            vectorDim,
            indexIndividual,
            index2 * populationDim + index_start_ind);
        if (simil > 0.7)
        {
            num_members += simil;
        }
    }

    // Displacement for population2
    displacement = islandNumber * population2Size;
    for (int index2 = displacement; index2 < population2Size + displacement; index2++)
    {
        float simil = cosine_similarity(
            population,
            population2,
            vectorDim,
            indexIndividual,
            index2 * populationDim + index_start_ind);
        if (simil > 0.7)
        {
            num_members += simil;
        }
    }

    if (num_members == 0)
    {
        return 1;
    }

    return num_members;
}

void init_end_iteration(
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

void train_step(
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
    float *fitness_array,
    float *fitness_spring,
    const int islandNumber,
    const int entityLabel)
{

    int start, end;
    start = islandNumber * population_size;
    end = (islandNumber + 1) * population_size;

    for (int index = start; index < end; index++)
    {
        mutate(
            population,
            population_offspring,
            index,
            populationDim);
    }

    // Fitness parent population
    for (int index = start; index < end; index++)
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
            metaDim,
            entityLabel);
    }
    // Fitness offspring population
    for (int index = start; index < end * 2; index++)
    {
        fitness_spring[index] = fitness_by_individual(
            population_offspring,
            docs,
            targets,
            meta,
            unknown_id,
            docs_count,
            index,
            populationDim,
            docsDim,
            docLen,
            metaDim,
            entityLabel);
    }
    // Selection
    // Fitness shared using cosine similarity
    for (int index = start; index < end; index++)
    {
        int num_members = individual_shared_fitness(
            index,
            population,
            population_offspring,
            populationDim,
            population_size,
            population_size * 2,
            islandNumber);
        fitness_array[index] = fitness_array[index] / num_members;
    }

    for (int index = islandNumber; index < end * 2; index++)
    {
        int num_members = individual_shared_fitness(
            index,
            population_offspring,
            population,
            populationDim,
            population_size * 2,
            population_size,
            islandNumber);
        fitness_spring[index] = fitness_spring[index] / num_members;
    }
}

void population_sort(
    float *population,
    float *population_offspring,
    float *fitness_array,
    float *fitness_spring,
    int population_size,
    int populationDim,
    int islandNumber)
{
    int initIteration, endIteration;
    initIteration = islandNumber * population_size;
    endIteration = (islandNumber + 1) * population_size;
    for (int index = 0; index < population_size; index++)
    {
        int index1 = index + population_size * islandNumber;
        if (index + 1 < population_size)
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
            if (fitness_array[index1] < fitness_spring[index1 * 2])
            {
                swap_fitness(
                    population,
                    fitness_array,
                    index1,
                    population_offspring,
                    fitness_spring,
                    index1 * 2,
                    populationDim);
            }

            if (fitness_array[index1] < fitness_spring[index1 * 2 + 1])
            {
                swap_fitness(
                    population,
                    fitness_array,
                    index1,
                    population_offspring,
                    fitness_spring,
                    index1 * 2 + 1,
                    populationDim);
            }
        }
    }
}

void train(
    float *population,
    float *fitness_array,
    float *population_offspring,
    float *fitness_spring,
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
    int numIslands,
    int entityLabel)
{

    for (size_t islandNumber = 0; islandNumber < numIslands; islandNumber++)
    {
        train_step(
            population,
            population_offspring,
            docs,
            targets,
            meta,
            unknown_id,
            docs_count,
            population_size / numIslands,
            populationDim,
            docsDim,
            docLen,
            metaDim,
            fitness_array,
            fitness_spring,
            islandNumber,
            ((entityLabel + islandNumber) % NUM_TAGS) + 1);
        population_sort(
            population,
            population_offspring,
            fitness_array,
            fitness_spring,
            population_size / numIslands,
            populationDim,
            islandNumber);
    }
}

void read_file(
    float ***docs,
    int **targets,
    float **meta,
    int docsDim,
    int docLen,
    int metaDim,
    int docs_count)
{
    // Read input file
    std::fstream file;
    file.open("../../../data/train/input.txt", std::ios::in);

    int i = 0;
    int j;
    int k;

    // Read line by line to get document. Each line is a document. Each document has 3 dimensions separated by a comma
    std::string document;
    while (getline(file, document))
    {
        // Read each dimension of the document
        std::string rows;
        std::stringstream ss(document);
        docs[i] = new float *[docLen];
        j = 0;
        while (getline(ss, rows, ' ') && j < docLen)
        {
            // Read each value of the dimension
            std::string value;
            std::stringstream ss2(rows);
            k = 0;
            docs[i][j] = new float[docsDim];
            while (getline(ss2, value, ',') && k < docsDim)
            {
                docs[i][j][k] = std::stof(value);
                k++;
            }
            j++;
        }
        i++;
    }

    file.close();
    // Read target file
    std::fstream file2;
    file2.open("../../../data/train/target.txt", ios::in);
    i = 0;
    // Read line by line to get document. Each line is a document. Each document has 3 dimensions separated by a comma
    while (getline(file2, document))
    {
        // Read each dimension of the document
        std::string rows;
        std::stringstream ss(document);
        targets[i] = new int[docLen];
        j = 0;
        // printf("document: %s ", document.c_str());
        while (getline(ss, rows, ' ') && j < docLen)
        {
            targets[i][j] = stoi(rows.c_str());
            j++;
        }
        i++;
    }

    file2.close();
    // Read meta file
    std::fstream file3;
    file3.open("../../../data/train/metadata.txt", ios::in);
    i = 0;
    // Read line by line to get document. Each line is a document. Each document has 3 dimensions separated by a comma
    while (getline(file3, document))
    {
        // Read each dimension of the document
        std::string rows;
        std::stringstream ss(document);
        meta[i] = new float[metaDim];
        j = 0;
        while (getline(ss, rows, ',') && j < metaDim)
        {
            meta[i][j] = stof(rows);
            j++;
        }
        i++;
    }

    file3.close();
}

int main(int argc, char *argv[])
{
    int i, tags = 1, tasks, iam;
    int rank;
    int population_size = 1200;
    int populationDim = 10;
    int docsDim = 3;
    int docLen = 100;
    int metaDim = 3;
    int docs_count = 200;
    int totalThreads;
    int unknown_id = UNK_ID;
    int max_iter = 1000;
    int tol = 0;
    int maxTolerance = 15;
    int minIter = 300;
    int numIslands = 8;
    int migrationRate = 10;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &iam);
    MPI_Comm_size(MPI_COMM_WORLD, &totalThreads);

    srand(42 + iam);
    // MPI_Scatterv
    float **population = new float *[population_size];
    float ***docs = new float **[docs_count];
    int **targets = new int *[docs_count];
    float **meta = new float *[docs_count];

    int start, end;
    init_end_iteration(
        iam,
        numIslands,
        totalThreads,
        start,
        end);

    int numIsladProcess = end - start;
    int populationPerIsland = population_size / numIslands;
    printf("iam %d Population per Island %d\n", iam, populationPerIsland);
    int populationPerProcess = populationPerIsland * numIsladProcess;
    printf("iam %d Population per process %d\n", iam, populationPerProcess);
    float *h_population = new float[population_size * populationDim];
    float *h_fitness = new float[population_size];
    float *s_population = new float[populationPerProcess * populationDim];
    float *s_fitness = new float[populationPerProcess];

    float *h_docs = new float[docs_count * docsDim * docLen];
    int *h_targets = new int[docs_count * docLen];
    float *h_meta = new float[docs_count * metaDim];

    int *sendcounts = new int[totalThreads];
    int *displs = new int[totalThreads];

    int *sendcountsFitness = new int[totalThreads];
    int *displsFitness = new int[totalThreads];

    for (int i = 0; i < totalThreads; i++)
    {
        int startProcess, endProcess;
        init_end_iteration(
            i,
            numIslands,
            totalThreads,
            startProcess,
            endProcess);
        // Fitness counts
        sendcountsFitness[i] = populationPerIsland * (endProcess - startProcess);
        displsFitness[i] = populationPerIsland * startProcess;
        // Population counts
        sendcounts[i] = sendcountsFitness[i] * populationDim;
        displs[i] = displsFitness[i] * populationDim;
    }

    if (iam == 0)
    {
        // Init population
        printf("Init population\n");
        for (int i = 0; i < population_size; i++)
        {
            population[i] = new float[populationDim];
            // Size of population
            population[i][0] = 1;
            population[i][1] = 0;
            // Entitiy type
            population[i][2] = (i % NUM_TAGS) + 1;
            // First random value
            population[i][3] = random_num(0, POS_SIZE + DEP_SIZE) + 1;
            for (int j = 4; j < populationDim; j++)
            {
                population[i][j] = 0;
            }
        }
        printf("Population initialized\n");
        // Read file to load docs
        read_file(
            docs,
            targets,
            meta,
            docsDim,
            docLen,
            metaDim,
            docs_count);
        printf("Docs read\n");
        for (int i = 0; i < population_size; i++)
        {
            h_fitness[i] = 0;
        }

        for (i = 1; i < totalThreads; i++)
        {
            // Flatten data

            // Copy population host
            for (int i = 0; i < population_size; i++)
            {
                for (int j = 0; j < populationDim; j++)
                {
                    h_population[i * populationDim + j] = population[start + i][j];
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
            printf("Docs copied\n");
            // Copy targets host
            for (int i = 0; i < docs_count; i++)
            {
                for (int j = 0; j < docLen; j++)
                {
                    h_targets[i * docLen + j] = targets[i][j];
                }
            }
            printf("Targets copied\n");
            // Copy meta host
            for (int i = 0; i < docs_count; i++)
            {
                for (int j = 0; j < metaDim; j++)
                {
                    h_meta[i * metaDim + j] = meta[i][j];
                }
            }

            printf("Meta copied\n");

            MPI_Send(
                h_docs,
                docs_count * docsDim * docLen,
                MPI_FLOAT,
                i,
                tags,
                MPI_COMM_WORLD);
            MPI_Send(
                h_targets,
                docs_count * docLen,
                MPI_INT,
                i,
                tags,
                MPI_COMM_WORLD);

            MPI_Send(
                h_meta,
                docs_count * metaDim,
                MPI_FLOAT,
                i,
                tags,
                MPI_COMM_WORLD);
        }

        // Print sendcounts and displs
        for (int i = 0; i < totalThreads; i++)
        {
            printf("iam %d sendcounts %d displs %d\n", iam, sendcounts[i], displs[i]);
        }
    }
    else
    {
        MPI_Recv(
            h_docs,
            docs_count * docsDim * docLen,
            MPI_FLOAT,
            0,
            tags,
            MPI_COMM_WORLD,
            &status);

        MPI_Recv(
            h_targets,
            docs_count * docLen,
            MPI_INT,
            0,
            tags,
            MPI_COMM_WORLD,
            &status);

        MPI_Recv(
            h_meta,
            docs_count * metaDim,
            MPI_FLOAT,
            0,
            tags,
            MPI_COMM_WORLD,
            &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatterv(
        h_population,
        sendcounts,
        displs,
        MPI_FLOAT,
        s_population,
        populationPerProcess * populationDim,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD);

    MPI_Scatterv(
        h_fitness,
        sendcountsFitness,
        displsFitness,
        MPI_FLOAT,
        s_fitness,
        populationPerProcess,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD);

    float *population2 = new float[populationPerProcess * populationDim * 2];
    float *fitness2 = new float[populationPerProcess * 2];
    float globalAvgFitness = 0;
    float globalCurAvgFitness = 0;
    float localAvgCurrentFitness = 0;
    bool stop = false;

    for (int iteration = 0; iteration < max_iter && !stop; iteration++)
    {
        train(
            s_population,
            s_fitness,
            population2,
            fitness2,
            h_docs,
            h_targets,
            h_meta,
            unknown_id,
            docs_count,
            populationPerProcess,
            populationDim,
            docsDim,
            docLen,
            metaDim,
            end - start,
            start);

        // Avg fitness
        for (int i = 0; i < populationPerProcess; i++)
        {
            localAvgCurrentFitness += s_fitness[i];
        }
        localAvgCurrentFitness /= populationPerProcess;
        MPI_Allreduce(
            &localAvgCurrentFitness,
            &globalAvgFitness,
            1,
            MPI_FLOAT,
            MPI_SUM,
            MPI_COMM_WORLD);

        globalAvgFitness /= totalThreads;
        if (globalAvgFitness > globalCurAvgFitness)
        {
            globalCurAvgFitness = globalAvgFitness;
            tol = 0;
            MPI_Gatherv(
                s_fitness,
                populationPerProcess,
                MPI_FLOAT,
                h_fitness,
                sendcountsFitness,
                displsFitness,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);

            MPI_Gatherv(
                s_population,
                populationPerProcess * populationDim,
                MPI_FLOAT,
                h_population,
                sendcounts,
                displs,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);

            // Copy data from h_population to population
            MPI_Barrier(MPI_COMM_WORLD);
            if (iam == 0)
            {
                for (int i = 0; i < population_size; i++)
                {
                    for (int j = 0; j < populationDim; j++)
                    {
                        population[i][j] = h_population[i * populationDim + j];
                    }
                }
                // Save in file individual with fitness > 0
                FILE *fpopulation = fopen("../../../data/rules/mpi/population.txt", "w");
                for (int i = 0; i < population_size; i++)
                {
                    if (h_fitness[i] > 0)
                    {
                        for (int j = 0; j < populationDim; j++)
                        {
                            fprintf(fpopulation, "%d ", (int)h_population[i * populationDim + j]);
                        }
                        fprintf(fpopulation, "\n");
                    }
                }
                fclose(fpopulation);
            }
        }
        else
        {
            if (tol > maxTolerance && iteration > minIter)
            {
                stop = true;
            }
            tol++;
        }

        // printf("iam %d iteration %d\n", iam, iteration);
        if (iteration % migrationRate == 0 && migrationRate > 0 && iteration > 1 && numIslands > 1)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            // Migrate best individuals to other islands and replace worst individuals
            for (int i = start; i < end; i++)
            {
                int bestIndex = (end - i - 1) * populationPerIsland;
                int worstIndex = (end - i) * populationPerIsland - 1;
                float individualFitness;
                float *individualMigration = new float[populationDim];
                float *individualMigrationRecv = new float[populationDim];
                for (int j = 0; j < populationDim; j++)
                {
                    individualMigration[j] = s_population[bestIndex * populationDim + j];
                }
                int neighborhood = (i + 1) % numIslands;
                if ((i + 1) < end)
                {
                    // Change best individual to neighbor in same process
                    float tempFitness = s_fitness[bestIndex];
                    while (s_population[worstIndex * populationDim + 2] != individualMigrationRecv[2] && worstIndex > bestIndex)
                    {
                        worstIndex--;
                    }
                    s_fitness[worstIndex] = tempFitness;
                    for (int j = 0; j < populationDim; j++)
                    {
                        s_population[worstIndex * populationDim + j] = individualMigration[j];
                    }
                }
                else
                {
                    // Send best individual to neighbor in different process
                    neighborhood = (iam + 1) % totalThreads;
                    float tempFitness = s_fitness[bestIndex];
                    // printf("iam %d sending to %d\n", iam, neighborhood);
                    MPI_Send(
                        &tempFitness,
                        1,
                        MPI_FLOAT,
                        neighborhood,
                        tags,
                        MPI_COMM_WORLD);

                    MPI_Send(
                        individualMigration,
                        populationDim,
                        MPI_FLOAT,
                        neighborhood,
                        tags,
                        MPI_COMM_WORLD);
                    // Receive worst individual from neighbor in different process
                    MPI_Barrier(MPI_COMM_WORLD);
                    int source = (iam - 1 + totalThreads) % totalThreads;
                    // printf("iam %d receiving from %d\n", iam, source);
                    MPI_Recv(
                        &individualFitness,
                        1,
                        MPI_FLOAT,
                        (iam - 1 + totalThreads) % totalThreads,
                        tags,
                        MPI_COMM_WORLD,
                        &status);

                    MPI_Recv(
                        individualMigrationRecv,
                        populationDim,
                        MPI_FLOAT,
                        source,
                        tags,
                        MPI_COMM_WORLD,
                        &status);

                    // Change for individual with same type
                    while (s_population[worstIndex * populationDim + 2] != individualMigrationRecv[2] && worstIndex > bestIndex)
                    {
                        worstIndex--;
                    }

                    // if (s_fitness[worstIndex] < individualFitness)
                    // if (s_fitness[worstIndex] == 0)
                    //{
                    s_fitness[worstIndex] = individualFitness;
                    printf("iam %d received fitness %.6f from %d\n", iam, individualFitness, source);
                    for (int j = 0; j < populationDim; j++)
                    {
                        s_population[worstIndex * populationDim + j] = individualMigrationRecv[j];
                        printf("%d ", (int)individualMigrationRecv[j]);
                    }
                    printf("\n");
                    //}
                }
            }
        }

        // print s_population
        // for (int i = 0; i < populationPerProcess; i++)
        // {
        //     for (int j = 0; j < populationDim; j++)
        //     {
        //         printf("%d ", (int)s_population[i * populationDim + j]);
        //     }
        //     printf("\n");
    }

    MPI_Gatherv(
        s_fitness,
        populationPerProcess,
        MPI_FLOAT,
        h_fitness,
        sendcountsFitness,
        displsFitness,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD);

    MPI_Gatherv(
        s_population,
        populationPerProcess * populationDim,
        MPI_FLOAT,
        h_population,
        sendcounts,
        displs,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD);

    // Copy data from h_population to population
    MPI_Barrier(MPI_COMM_WORLD);
    if (iam == 0)
    {

        // for (int i = 0; i < population_size; i++)
        // {
        //     for (int j = 0; j < populationDim; j++)
        //     {
        //         population[i][j] = h_population[i * populationDim + j];
        //     }
        // }
        printf("\n");
        // Save in file individual with fitness > 0
        // FILE *fpopulation = fopen("population.txt", "w");
        for (int i = 0; i < population_size; i++)
        {
            printf("%.6f ", h_fitness[i]);
            // if (h_fitness[i] > 0)
            // {
            //     for (int j = 0; j < populationDim; j++)
            //     {
            //         fprintf(fpopulation, "%d ", (int)h_population[i * populationDim + j]);
            //     }
            //     fprintf(fpopulation, "\n");
            // }
        }
        printf("\n");
        // print population
        // for (int i = 0; i < population_size; i++)
        // {
        //     for (int j = 0; j < populationDim; j++)
        //     {
        //         printf("%d  ", (int)population[i][j]);
        //     }
        //     printf("\n");
        // }
    }
    MPI_Finalize();
    return 0;
}
