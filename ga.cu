#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <tuple>
#include <vector>
#include <iterator>
#include <map>
#include <list>
#include <numeric>
#include <cstdlib>
#include <algorithm>
#include "hdf5.h"
#include "ga.cuh"
#include <omp.h>
#include <sys/time.h>
#include <fstream>

using namespace std;

// Function to generate random numbers in given range
__host__ int random_num(int start, int end)
{
    int range = (end - start) + 1;
    int random_int = range > 0 ? start + (rand() % range) : start;
    return random_int;
}

// Class representing individual in population
class Individual
{
public:
    Rule chromosome;
    float fitness;
    vector<Document *> doc_list;
    int docSize;

    Individual()
    {
        this->fitness = 0;
        this->docSize = 0;
    }

    Individual(Rule chromosome, Document doc_list[], int docSize)
    {
        this->chromosome = chromosome;
        this->fitness = chromosome.fitness(doc_list, docSize);
        this->docSize = docSize;
    }
    Individual(Rule chromosome, vector<Document *> doc_list)
    {
        this->chromosome = chromosome;
        this->fitness = chromosome.fitness(doc_list);
        this->doc_list = doc_list;
        this->docSize = doc_list.size();
    }

    Individual mate(Individual parent2, float mutation_rate, float crossover_rate, bool simple);
    Rule mutation(Rule offspring, int index_chomosome, int remove_operation, std::vector<std::string> operation_list);
    Rule mutation(Rule offspring, int index_chomosome);
    Rule mutation(Rule offspring, int index_chomosome, string operationList[]);
    float getFitness() { return fitness; }

    bool operator<(Individual &other)
    {
        return this->fitness < other.fitness;
    }

    bool operator>(Individual &other)
    {
        return this->fitness > other.fitness;
    }

    bool operator==(Individual &other)
    {
        return this->chromosome == other.chromosome;
    }
};

// Perform mating and produce new offspring
Individual Individual::mate(
    Individual par2, float mutation_rate = 0.1,
    float crossover_rate = 0.5, bool simple = true)
{
    // chromosome for offspring
    Rule child_chromosome(this->chromosome.tag, this->chromosome.token_list);

    // int len = chromosome.token_list.size();
    int i;
    if (simple)
    {
        // random probability
        float p = (float)random_num(0, 100) / 100.0;
        if (p <= mutation_rate)
        {
            i = random_num(0, this->chromosome.token_list.size() - 1);
            child_chromosome = mutation(child_chromosome, i);
        }
        else if (p <= (mutation_rate + crossover_rate))
        {
            i = random_num(0, min(this->chromosome.token_list.size() - 1, par2.chromosome.token_list.size() - 1));
            int j;
            int sizeChromosome = child_chromosome.token_list.size();
            for (j = i; j < par2.chromosome.token_list.size(); j++)
            {
                if (j >= sizeChromosome)
                {
                    child_chromosome.token_list.push_back(par2.chromosome.token_list[j]);
                }
                else
                {
                    child_chromosome.token_list[j] = par2.chromosome.token_list[j];
                }
            }
            // Delete the rest of the tokens
            while (j < child_chromosome.token_list.size())
            {
                child_chromosome.token_list.pop_back();
            }
        }
    }
    else
    {
        for (i = 0; i < child_chromosome.token_list.size(); i++)
        {
            // random probability
            float p = (float)random_num(0, 100) / 100.0;

            // if prob is between 0.45 and 0.90, insert
            // gene from parent 2
            if (p <= 0.1)
            {
                child_chromosome = mutation(child_chromosome, i);
            }
            else if (p <= (mutation_rate + crossover_rate) && i < par2.chromosome.token_list.size())
            {
                child_chromosome.token_list[i] = par2.chromosome.token_list[i];
            }
            // otherwise insert random gene(mutate),
            // for maintaining diversity
        }
    }

    // create new Individual(offspring) using
    // generated chromosome for offspring
    return Individual(child_chromosome, doc_list);
};

// Overloading < operator
bool operator<(const Individual &ind1, const Individual &ind2)
{
    return ind1.fitness < ind2.fitness;
}

/** Print values of a map */
template <typename K, typename V>
void print_map(std::map<K, V> const &m)
{
    for (auto const &pair : m)
    {
        std::cout << "{" << pair.first << ": " << pair.second << "}\n";
    }
}

/** Print values of a vector */
template <typename T>
void print_vector(std::vector<T> const &v)
{
    for (auto const &elem : v)
    {
        std::cout << elem << ' ';
    }
    std::cout << '\n';
}

/** Print values of a tuple */
template <typename T, typename U>
void print_tuple(std::tuple<T, U> const &t)
{
    std::cout << "{" << get<0>(t) << ", " << get<1>(t) << '\n';
}

/** Print values of a vector tuple */
template <typename T, typename U>
void print_vector_tuple(std::vector<std::tuple<T, U>> const &v);

/** Calculate Jaccard similarity between two vectors */
double jaccard_similarity(vector<int> v1, vector<int> v2)
{
    int intersection = 0;
    int union_ = 0;
    for (int i = 0; i < v1.size(); i++)
    {
        if (v1[i] == 1 && v2[i] == 1)
        {
            intersection++;
        }
        if (v1[i] == 1 || v2[i] == 1)
        {
            union_++;
        }
    }
    if (union_ == 0)
    {
        return -1;
    }
    printf("%d/%d\n", intersection, union_);
    return (double)intersection / (double)union_;
}

double jaccard_similarity(std::vector<int *> v1, std::vector<int *> v2)
{
    int intersection = 0;
    int union_ = 0;
    for (int i = 0; i < v1.size(); i++)
    {
        if (v1[i] == equalint && v2[i] == equalint)
        {
            intersection++;
        }
        if (v1[i] == equalint || v2[i] == equalint)
        {
            union_++;
        }
    }
    if (union_ == 0)
    {
        return -1;
    }
    return (double)intersection / (double)union_;
}

__device__ __host__ double jaccard_similarity(int v1[], int v2[], int sizeVector)
{
    int intersection = 0;
    int union_ = 0;
    for (int i = 0; i < sizeVector; i++)
    {
        if (v1[i] == 1 && v2[i] == 1)
        {
            intersection++;
        }
        if (v1[i] == 1 || v2[i] == 1)
        {
            union_++;
        }
    }
    if (union_ == 0)
    {
        return 1;
    }
    return (double)intersection / (double)union_;
}

__device__ __host__ float jaccard_similarity(
    int v1[], int v2[], int sizeVector,
    unsigned int &intersection, unsigned int &iunion)
{

    for (int i = 0; i < sizeVector; i++)
    {
        if (v1[i] == 1 && v2[i] == 1)
        {
            intersection += 1;
        }
        if (v1[i] == 1 || v2[i] == 1)
        {
            iunion += 1;
        }
    }

    if (iunion == 0)
    {
        return 1;
    }
    return (float)(intersection / iunion);
}

/**
 * Returns span list of entity tuples in a document regard rule list
 */
std::vector<std::tuple<int, int>> Rule::get_span_list(Document *doc)
{
    int doc_size = doc->tokens.size();
    std::vector<std::tuple<int, int>> span_list;
    std::vector<RuleToken> rules = token_list;
    for (int i = 0; i < doc_size; i++)
    {
        std::map<std::string, std::string> *token = &doc->tokens[i];
        if (doc_size - i < rules.size())
        {
            break;
        }
        int span = i;
        int start = -1;
        int end = -1;
        std::vector<std::tuple<int, int>> temp_span_list;
        bool copy_temp_span_list = true;
        bool anyIncludes = false;
        // // cout << span << " " << doc_size << '\n';
        for (int j = 0; j < rules.size(); j++)
        {
            RuleToken *ruleToken = &rules[j];
            std::string *prop_key = &get<0>(ruleToken->pos);
            std::string *prop_value = &get<1>(ruleToken->pos);
            bool *is_include = &ruleToken->is_include;
            // cout <<prop_key << " " << prop_value << " " << token.at(prop_key) << '\n';
            if (token->at(prop_key->c_str()) == prop_value->c_str())
            {

                if (is_include)
                {
                    anyIncludes = true;
                    if (start == -1)
                    {
                        start = span;
                        end = span;
                    }
                    else
                    {
                        end = span;
                    }
                }
                else if (start != -1)
                {
                    temp_span_list.push_back(make_tuple(start, end));
                    start = -1;
                    end = -1;
                }
                span++;
                if (span >= doc_size)
                {
                    break;
                }
                else
                {
                    token = &doc->tokens[span];
                }
            }
            else
            {
                copy_temp_span_list = false;
                break;
            }
        }

        if (start != -1)
        {
            temp_span_list.push_back(make_tuple(start, end));
        }
        if (copy_temp_span_list && anyIncludes)
        {
            span_list.insert(span_list.end(), temp_span_list.begin(), temp_span_list.end());
        }
    }
    return span_list;
}

__device__ __host__ int get_span_list(rule_t rule, document_t document, int *spanList, unsigned int numTokens)
{
    int entity_count = 0;
    // printf("tokenSize: %d \n", rule.tokenSize);
    // printf("numTokens: %d \n", numTokens);
    // printf("POS_LIST_SIZE: %d \n", POS_LIST_SIZE);
    for (int i = 0; i < numTokens; i++)
    {

        if (numTokens - i >= rule.tokenSize)
        {

            int span = i;
            int tempStart = -1;
            int tempEnd = -1;
            int j = 0;
            int antEntityCount = entity_count;
            bool allIncludes = true;

            for (j = 0; j < rule.tokenSize; j++)
            {
                if (allIncludes)
                {
                    bool includesAnt = false;
                    int featureKey = rule.tokens[j * POS_LIST_SIZE];
                    int featureValue = rule.tokens[j * POS_LIST_SIZE + 1];
                    bool is_include = rule.tokens[j * POS_LIST_SIZE + 2];
                    // printf("Span: %d, Feature: %d, DOC VAL: %d, Value: %d, Include: %d\n", span, featureKey, document.tokens[span * POS_LIST_SIZE + featureKey], featureValue, is_include);
                    if (document.tokens[span * POS_LIST_SIZE + featureKey] == featureValue)
                    {
                        if (is_include == 1)
                        {
                            if (includesAnt != 1)
                            {
                                tempStart = span;
                            }
                            // printf("LIMIT: %d, %d\n", span, (int)(numTokens / POS_LIST_SIZE) - 1);
                            //  limit document
                            if (span == numTokens - 1)
                            {
                                tempEnd = span;
                                spanList[entity_count * 2] = tempStart;
                                spanList[entity_count * 2 + 1] = tempEnd;
                                entity_count++;
                                // printf("1 Count: %d, Start: %d, End: %d\n", entity_count, tempStart, tempEnd);
                            }
                            // limit rule
                            else if (j == rule.tokenSize - 1)
                            {
                                tempEnd = span;
                                spanList[entity_count * 2] = tempStart;
                                spanList[entity_count * 2 + 1] = tempEnd;
                                entity_count++;
                                // printf("2 Count: %d, Start: %d, End: %d\n", entity_count, tempStart, tempEnd);
                            }
                        }
                        else if (includesAnt != 1)
                        {
                            tempStart = span;
                        }
                        else if (includesAnt == 1)
                        {
                            tempEnd = span;
                            spanList[entity_count * 2] = tempStart;
                            spanList[entity_count * 2 + 1] = tempEnd;
                            // printf("3 Count: %d, Start: %d, End: %d\n", entity_count, tempStart, tempEnd);
                            entity_count++;
                        }
                        includesAnt = is_include;
                        span++;
                        if (span >= numTokens)
                        {
                            allIncludes = false;
                        }
                    }
                    else
                    {
                        allIncludes = false;
                    }
                }
            }
            if (!allIncludes)
            {
                entity_count = antEntityCount;
            }
        }
    }
    return entity_count;
}

__device__ __host__ float fitness(rule_t rule, document_t *documents, unsigned int nDocs)
{
    unsigned int intersection = 0;
    unsigned int union_ = 0;
    int tagId = rule.entityId;
    unsigned int totalRetrive = 0;
    for (int i = 0; i < nDocs; i++)
    {
        // fill relevant_doc with 1 if the token is in the span_list
        int relevantDocument[172];
        document_t document = documents[i];
        int docSize = document.tokenSize / POS_LIST_SIZE;
        for (int j = 0; j < 172; j++)
        {
            relevantDocument[j] = 0;
        }
        for (int j = 0; j < document.entitySize; j += 3)
        {
            // printf("Entity:%d Tag:%d\n", document.entities[j], tagId);
            if (document.entities[j] == tagId)
            {
                int start = document.entities[j + 1];
                int end = document.entities[j + 2];
                for (int k = start; k <= end; k++)
                {
                    relevantDocument[k] = 1;
                }
            }
        }
        // fill retrive_doc with 1 if the token is in the span_list
        int retriveDocument[172];
        for (int j = 0; j < 172; j++)
        {
            retriveDocument[j] = 0;
        }
        int spanList[360];
        int retriveEntitySize = get_span_list(rule, document, spanList, docSize);
        totalRetrive += retriveEntitySize;
        if (retriveEntitySize > 0)
        {
            // printf("retriveEntitySize: %d rule size:%d\n", retriveEntitySize, rule.tokenSize);
            for (int j = 0; j < retriveEntitySize; j++)
            {
                int start = spanList[j * 2];
                int end = spanList[j * 2 + 1];
                // printf("Start: %d, End: %d, docSize:%d doc%d\n", start, end, docSize, i);
                for (int k = start; k <= end; k++)
                {
                    retriveDocument[k] = 1;
                }
            }
        }
        // for (int j = 0; j < document.tokenSize / POS_LIST_SIZE; j++)
        // {
        //     printf("%d %d\n", relevantDocument[j], retriveDocument[j]);
        // }
        jaccard_similarity(retriveDocument, relevantDocument, docSize, intersection, union_);
        // if (retriveEntitySize > 0)
        // {
        //     printf("doc %d: size:%d %d/%d ", i, docSize, intersection, union_);
        //     for (int j = 0; j < docSize; j++)
        //     {
        //         printf("token: %d %d-%d, ", j, relevantDocument[j], retriveDocument[j]);
        //     }
        //     printf("\n");
        // }
    }

    // if (totalRetrive > 0)
    // {
    //     printf("Score %f: %d/%d\n", ((float)intersection / union_) * log(intersection + 1), intersection, union_);
    // }

    if (totalRetrive < 1)
        return 0.0;
    if (union_ < 1)
        return 0.0;

    // printf("%d/%d=%f\n", intersection, union_, ((float)intersection / union_) * log(intersection + 1));
    return ((float)intersection / union_);
}
/** Create fitness function to evaluate rule over doc_list by for a tag*/
float Rule::fitness(
    vector<Document *> doc_list)
{
    float score = 0.0;
    int score_count = 0;
    int docSize = doc_list.size();
#pragma omp parallel for reduction(+ \
                                   : score, score_count)
    for (int i = 0; i < docSize; i++)
    {
        Document *doc = doc_list[i];
        auto spanPointer = doc->entities.find(tag);
        std::vector<std::tuple<int, int>> span_list;
        if (spanPointer != doc->entities.end())
        {
            span_list = spanPointer->second;
        }
        int total_tokens = doc->tokens.size();
        // fill relevant_doc with 1 if the token is in the span_list
        int relevant_doc[total_tokens];
        if (span_list.size() > 0)
        {
            for (int j = 0; j < span_list.size(); j++)
            {
                std::tuple<int, int> span = span_list[j];
                int start = get<0>(span);
                int end = get<1>(span) + 1;
                for (int k = 0; k < total_tokens; k++)
                {
                    if (k >= start && k < end)
                    {
                        relevant_doc[k] = 1;
                    }
                    else
                    {
                        relevant_doc[k] = 0;
                    }
                }
            }
        }

        // fill retrive_doc with 1 if the token is in the span_rule_list
        int retrive_doc[total_tokens];
        std::vector<std::tuple<int, int>> span_rule_list = get_span_list(doc);
        if (span_rule_list.size() > 0)
        {
            for (int j = 0; j < span_rule_list.size(); j++)
            {
                std::tuple<int, int> span = span_rule_list[j];
                int start = get<0>(span);
                int end = get<1>(span) + 1;
                for (int k = 0; k < total_tokens; k++)
                {
                    if (k >= start && k < end)
                    {
                        retrive_doc[k] = 1;
                    }
                    else
                    {
                        retrive_doc[k] = 0;
                    }
                }
            }
        }
        int docSize = sizeof(relevant_doc) / sizeof(relevant_doc[0]);
        float jaccard_score = jaccard_similarity(relevant_doc, retrive_doc, docSize);
        if (jaccard_score != -1)
        {
            score += jaccard_score - (docSize / 1000);
            score_count++;
        }
    }
    if (score_count != 0)
    {
        score /= score_count;
    }
    else
    {
        score = -1;
    }
    return score;
}

/** create function to random choice items from list */
template <typename T>
std::vector<T> random_choice(std::vector<T> v1, int n_item)
{
    std::vector<T> selectedList;
    for (int i = 0; i < n_item; i++)
    {
        int index = rand() % v1.size();
        selectedList[i] = v1[index];
        v1.erase(v1.begin() + index);
    }
    return selectedList;
}
/** create function to random choice a item from list */
template <typename T>
T random_choice(std::vector<T> v1)
{
    int index = rand() % v1.size();
    return v1[index];
}

/** Create function to random choice a item from array */
template <typename T>
T random_choice(T v1[])
{
    int n_item = sizeof(v1) / sizeof(v1[0]);
    int index = rand() % n_item;
    return v1[index];
}

/** Create mutation function to mutate a rule regard tag*/
Rule Individual::mutation(
    Rule offspring, int index_chomosome, int remove_operation,
    std::vector<std::string> operation_list = {"add_prev", "add_after", "change_gen", "rem_prev", "rem_after"})
{
    // std::vector<std::string> operation_list = operationList;
    if (remove_operation > -1)
    {
        operation_list.erase(operation_list.begin() + remove_operation);
    }
    std::string operation = random_choice(operation_list);
    std::vector<std::string> representation_list = {"dep_", "lemma_", "pos_"};
    std::string representation;
    std::string tag = offspring.tag;
    // cout << "operation: " << operation << '\n';
    if (operation == "add_prev")
    {
        RuleToken ruleToken = offspring.token_list[0];
        int doc_id = ruleToken.document_id;
        int pos_id = ruleToken.token_position;
        Document *doc = doc_list[doc_id];
        if (pos_id == 0)
        {
            operation = "change_gen";
        }
        else
        {
            representation = random_choice(representation_list);
            int new_pos = pos_id - 1;
            std::string val_representation = doc->tokens[new_pos][representation];
            bool add_tag = false;
            std::vector<std::tuple<int, int>> span_list = doc->entities.at(tag);
            for (int i = 0; i < span_list.size(); i++)
            {
                std::tuple<int, int> span = span_list[i];
                int start = get<0>(span);
                int end = get<1>(span) + 1;
                if (new_pos >= start && new_pos <= end)
                {
                    add_tag = true;
                    break;
                }
            }
            RuleToken newRuleToken = RuleToken({representation, val_representation}, add_tag, doc_id, new_pos);
            offspring.token_list.insert(offspring.token_list.begin(), newRuleToken);
        }
    }
    else if (operation == "add_after")
    {
        RuleToken ruleToken = offspring.token_list[offspring.token_list.size() - 1];
        int doc_id = ruleToken.document_id;
        int pos_id = ruleToken.token_position;
        Document *doc = doc_list[doc_id];

        if (pos_id + 1 >= doc->tokens.size())
        {
            operation = "change_gen";
        }
        else
        {
            representation = random_choice(representation_list);
            int new_pos = pos_id + 1;
            std::string val_representation = doc->tokens[new_pos][representation];
            bool add_tag = false;
            std::vector<std::tuple<int, int>> span_list = doc->entities.at(tag);
            for (int i = 0; i < span_list.size(); i++)
            {
                std::tuple<int, int> span = span_list[i];
                int start = get<0>(span);
                int end = get<1>(span) + 1;
                if (new_pos >= start && new_pos <= end)
                {
                    add_tag = true;
                    break;
                }
            }
            RuleToken newRuleToken = RuleToken({representation, val_representation}, add_tag, doc_id, new_pos);
            offspring.token_list.push_back(newRuleToken);
        }
    }
    else if (operation == "rem_prev")
    {
        if (offspring.token_list.size() == 1)
        {
            operation = "change_gen";
        }
        else
        {
            offspring.token_list.erase(offspring.token_list.begin());
        }
    }
    else if (operation == "rem_after")
    {
        if (offspring.token_list.size() == 1)
        {
            operation = "change_gen";
        }
        else
        {
            offspring.token_list.erase(offspring.token_list.end() - 1);
        }
    }

    if (operation == "change_gen")
    {
        RuleToken ruleToken = offspring.token_list[index_chomosome];
        int doc_id = ruleToken.document_id;
        int pos_id = ruleToken.token_position;
        Document *doc = doc_list[doc_id];

        std::string currentRepresentation = get<0>(ruleToken.pos);
        representation_list.erase(
            remove(representation_list.begin(), representation_list.end(), currentRepresentation),
            representation_list.end());
        representation = random_choice(representation_list);
        std::string val_representation = doc->tokens[pos_id][representation];
        RuleToken newRuleToken = RuleToken(
            {representation, val_representation}, ruleToken.is_include, doc_id, pos_id);
        offspring.token_list[index_chomosome] = newRuleToken;
    }
    return offspring;
};

Rule Individual::mutation(Rule offspring, int index_chomosome)
{
    return mutation(offspring, index_chomosome, -1);
}

RuleToken::RuleToken(tuple<string, string> pos, bool is_include, int document_id, int token_position)
{
    this->pos = pos;
    this->is_include = is_include;
    this->document_id = document_id;
    this->token_position = token_position;
}

/**Print object*/
void RuleToken::_print()
{
    cout << "(" << get<0>(pos) << "," << get<1>(pos) << ")";
    if (is_include)
        cout << " + ";
    else
        cout << " - ";
    cout << "(" << document_id << "," << token_position << ")\n";
}

/**
 * Class Rule
 * Representation for a PoS rule
 */

Rule::Rule() {}

Rule::Rule(string tag)
{
    this->tag = tag;
}

Rule::Rule(string tag, vector<RuleToken> token_list)
{
    this->tag = tag;
    this->token_list = token_list;
}

void Rule::add_token(tuple<string, string> pos, bool is_include, int document_id, int token_position)
{
    this->token_list.push_back(RuleToken(pos, is_include, document_id, token_position));
}

void Rule::insert_token(tuple<string, string> pos, bool is_include, int document_id, int token_position)
{
    this->token_list.insert(this->token_list.begin(), RuleToken(pos, is_include, document_id, token_position));
}

/** Print values of a vector tuple */
template <typename T, typename U>
void print_vector_tuple(std::vector<std::tuple<T, U>> const &v)
{
    for (auto const &elem : v)
    {
        std::cout << '{' << get<0>(elem) << ", " << get<1>(elem) << "}\n";
    }
    std::cout << '\n';
}

/**Read String array from H5 file in c++*/
std::map<int, std::string> read_string(hid_t hfile, const char *dset_name)
{
    hid_t dset, space_id, memtype;
    int i;
    hsize_t dims[1];

    dset = H5Dopen(hfile, dset_name, H5P_DEFAULT);
    // get the storage size and space_id
    H5Dget_storage_size(dset);
    space_id = H5Dget_space(dset);
    // set the memory type to be a string
    memtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(memtype, H5T_VARIABLE);
    H5Tset_strpad(memtype, H5T_STR_NULLTERM);
    H5Tset_cset(memtype, H5T_CSET_UTF8);
    // get the size of the dataset
    H5Sget_simple_extent_dims(space_id, dims, NULL);
    // read string from dataset
    int n = dims[0];
    char *str_array[n];
    H5Dread(dset, memtype, space_id, H5S_ALL, H5P_DEFAULT, str_array);

    H5Sclose(space_id);
    H5Tclose(memtype);
    H5Dclose(dset);
    // Char array to map string
    std::map<int, std::string> str_map;
    for (i = 0; i < n; i++)
    {
        str_map[i] = str_array[i];
    }
    return str_map;
}

#define NX_SUB 3 /* hyperslab dimensions */
#define NY_SUB 4
#define NX 50 /* output buffer dimensions */
#define NY 172
#define NZ 172
#define RANK 2
#define RANK_OUT 3
#define N_EXAMPLE 2111

/**Read data from H5 file to load Document*/
herr_t
op_func(hid_t loc_id, const char *name, const H5L_info_t *info,
        void *operator_data);

map<string, vector<tuple<int, int>>> getEntityList(string tagList[])
{
    map<string, vector<tuple<int, int>>> entityList;

    int span_start = -1;
    int span_end = -1;
    string prev_tag = "";

    // Documents
    string prefix;
    int j;
    for (j = 0; j < NY; j++)
    {
        prefix = "O";
        string tag = "";
        if (tagList[j] == "-PAD")
        {
            break;
        }
        if (tagList[j] != "O")
        {
            prefix = tagList[j][0];
            tag = tagList[j].substr(2);
        }

        if (prefix == "B")
        {
            span_start = j;
            span_end = j;
        }
        else if (prefix == "I")
        {
            span_end = j;
        }
        else if (span_start != -1)
        {
            entityList[prev_tag].push_back(make_tuple(span_start, span_end));
            span_start = -1;
            span_end = -1;
        }
        else
        {
            span_start = -1;
            span_end = -1;
        }

        prev_tag = tag;
    }
    if (prefix == "B")
    {
        entityList[prev_tag].push_back(make_tuple(span_start, span_end));
    }
    else if (prefix == "I")
    {
        span_end = j - 1;
        entityList[prev_tag].push_back(make_tuple(span_start, span_end));
    }

    return entityList;
}

vector<Document *> generateDocumentList(
    map<string, int[NX][NY]> input_map,
    map<string, map<int, string>> vocab_map,
    string tagList[NX][NY],
    string *inputPos,
    int n)
{
    vector<Document *> doc_list;
    int j;
    // Iterate over documents
    for (int i = 0; i < n; i++)
    {
        Document *doc = new Document();
        vector<map<string, string>> token_list;
        // Iterate over tokens
        for (j = 0; j < NY; j++)
        {
            map<string, string> token_map;
            // Iterate over tokens
            bool is_valid = false;
            for (int k = 0; k < 3; k++)
            {
                string pos = inputPos[k];
                int val_id = input_map[pos][i][j];
                if (val_id == 0)
                {
                    is_valid = false;
                    break;
                }
                string val = vocab_map[pos][val_id];
                token_map[pos] = val;
                is_valid = true;
            }
            if (is_valid)
            {
                token_list.push_back(token_map);
            }
        }
        doc->tokens = token_list;
        doc->entities = getEntityList(tagList[i]);
        doc_list.push_back(doc);
    }

    return doc_list;
}

/**Read dataset from H5 File by offset and count*/
void read_dataset(hid_t file_id, const char *name, hid_t type_id, const hsize_t *offset, const hsize_t *count, void *buf)
{
    hid_t dataset_id, dataspace_id, memspace_id;

    dataset_id = H5Dopen2(file_id, name, H5P_DEFAULT);
    dataspace_id = H5Dget_space(dataset_id);
    int rank = H5Sget_simple_extent_ndims(dataspace_id);
    memspace_id = H5Screate_simple(rank, count, NULL);
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    H5Dread(dataset_id, type_id, memspace_id, dataspace_id, H5P_DEFAULT, buf);
    H5Sclose(memspace_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
}

vector<Document *> read_data(
    std::string file_name,
    document_t *&document_pnt,
    int *&featureSize,
    map<int, string> &id2tagSimple,
    map<string, map<int, string>> &vocab_map,
    map<string, map<string, int>> &vocab2id_map)
{
    hid_t file_id;
    int i, j, k;
    vector<Document *> doc_list;
    // TODO Export Vocab2Id
    map<int, string> id2tag_map;
    map<string, int> tag2idSimple;
    map<string, int[NX][NY]> input_map;
    map<string, int[NX][NY]> tag_map;

    string inputPos[POS_LIST_SIZE] = {"dep_", "lemma_", "pos_"};
    string tagList[2] = {"predict", "real"};
    featureSize = (int *)malloc(sizeof(int) * POS_LIST_SIZE);

    file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    printf("Objects in root group:\n");
    H5Literate(file_id, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, op_func, NULL);

    // Read id2tag
    id2tag_map = read_string(file_id, "id2tag");
    j = 0;
    for (i = 0; i < id2tag_map.size(); i++)
    {
        string tag = id2tag_map[i] == "O" ? "O" : id2tag_map[i].substr(2);
        if (tag2idSimple.find(tag) == tag2idSimple.end())
        {
            tag2idSimple[tag] = j;
            id2tagSimple[j] = tag;
            j++;
        }
    }

    int *idTagArray = (int *)malloc(sizeof(int) * id2tagSimple.size());
    for (i = 0; i < id2tagSimple.size(); i++)
    {
        idTagArray[i] = tag2idSimple[id2tagSimple[i]];
    }

    // Read Vocab
    for (i = 0; i < POS_LIST_SIZE; i++)
    {
        string dataset_name = "vocab_" + inputPos[i];
        vocab_map[inputPos[i]] = read_string(file_id, dataset_name.c_str());
        map<string, int> mapId;
        for (j = 0; j < vocab_map[inputPos[i]].size(); j++)
        {
            string word = vocab_map[inputPos[i]][j];
            mapId[word] = j;
        }
        vocab2id_map[inputPos[i]] = mapId;
        featureSize[i] = vocab_map[inputPos[i]].size();
    }

    for (i = 0; i < (1 + N_EXAMPLE / NX); i++)
    {
        int start_position = i * NX;
        int count_items = min(NX, N_EXAMPLE - start_position);
        hsize_t count_out[2];
        hsize_t offset_out[2];
        int data_out[NX][NY];
        offset_out[0] = start_position;
        offset_out[1] = 0;
        count_out[0] = count_items;
        count_out[1] = NY;
        // Read input data
        for (j = 0; j < POS_LIST_SIZE; j++)
        {
            string dataset_name = "input_" + inputPos[j];
            read_dataset(file_id, dataset_name.c_str(), H5T_NATIVE_INT, offset_out, count_out, data_out);
            for (k = 0; k < NX; k++)
            {
                for (int l = 0; l < NY; l++)
                {
                    input_map[inputPos[j]][k][l] = data_out[k][l];
                }
            }
        }
        // Read tag data
        for (j = 0; j < 2; j++)
        {
            string dataset_name = tagList[j];
            read_dataset(file_id, dataset_name.c_str(), H5T_NATIVE_INT, offset_out, count_out, data_out);
            for (k = 0; k < NX; k++)
            {
                for (int l = 0; l < NY; l++)
                {
                    tag_map[dataset_name][k][l] = data_out[k][l];
                }
            }
        }

        string tagStrList[NX][NY];
        for (j = 0; j < NX; j++)
        {
            for (k = 0; k < NY; k++)
            {
                tagStrList[j][k] = id2tag_map[tag_map["real"][j][k]];
            }
        }
        vector<Document *> doc_vector = generateDocumentList(input_map, vocab_map, tagStrList, inputPos, count_items);
        doc_list.insert(end(doc_list), begin(doc_vector), end(doc_vector));
    }
    H5Fclose(file_id);

    // Map data to document structure
    document_pnt = (document_t *)malloc(sizeof(document_t) * doc_list.size());
    int n = doc_list.size();
    // n = 90;
    for (int i = 0; i < n; i++)
    {
        Document *doc = doc_list[i];
        int tokenSize = doc->size() * POS_LIST_SIZE;
        document_pnt[i].tokens = (int *)malloc(sizeof(int) * tokenSize);
        document_pnt[i].tokenSize = tokenSize;
        for (int j = 0; j < doc->tokens.size(); j++)
        {
            for (int k = 0; k < POS_LIST_SIZE; k++)
            {
                string pos = POS_LIST[k];
                string posValue = doc->tokens[j][pos];
                document_pnt[i].tokens[j * POS_LIST_SIZE + k] = vocab2id_map[pos][posValue];
            }
        }
        int entitySize = 0;
        for (auto const &entity : doc->entities)
        {
            entitySize += entity.second.size() * 3;
        }
        document_pnt[i].entitySize = entitySize;
        document_pnt[i].entities = (int *)malloc(sizeof(int) * entitySize);

        int j = 0;
        for (auto const &entity : doc->entities)
        {
            string tagName = entity.first;

            for (int k = 0; k < entity.second.size(); k++)
            {
                int start = get<0>(entity.second[k]);
                int end = get<1>(entity.second[k]);
                document_pnt[i].entities[j * 3] = tag2idSimple[tagName];

                document_pnt[i].entities[j * 3 + 1] = start;
                document_pnt[i].entities[j * 3 + 2] = end;
                j++;
            }
        }
    }
    return doc_list;
}

herr_t op_func(hid_t loc_id, const char *name, const H5L_info_t *info,
               void *operator_data)
{
    H5O_info_t infobuf;

    /*
     * Get type of the object and display its name and type.
     * The name of the object is passed to this function by
     * the Library.
     */
    H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT);
    switch (infobuf.type)
    {
    case H5O_TYPE_GROUP:
        printf("  Group: %s\n", name);
        break;
    case H5O_TYPE_DATASET:
        printf("  Dataset: %s\n", name);
        break;
    case H5O_TYPE_NAMED_DATATYPE:
        printf("  Datatype: %s\n", name);
        break;
    default:
        printf("  Unknown: %s\n", name);
    }

    return 0;
}

#include <unistd.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, unsigned int n, unsigned int totalThreads, curandState_t *states)
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

__device__ void createToken(
    rule_t *child, int chromosoneId,
    int *featureSize, curandState_t *state)
{
    int feature = curand(state) % POS_LIST_SIZE;
    child->tokens[chromosoneId * 3] = feature;
    child->tokens[chromosoneId * 3 + 1] = curand(state) % featureSize[feature];
    child->tokens[chromosoneId * 3 + 2] = child->tokenSize > 1 ? curand(state) % 2 : 1;
}

__device__ void mutation(rule_t *parent, rule_t *child, int *featureSize, curandState_t *state)
{
    child->entityId = parent->entityId;
    child->tokenSize = parent->tokenSize;
    int numOperators = parent->tokenSize < 2 ? 3 : 5;
    // 0: mutatation, 1: add token at begin, 2: add token at end, 3: remove token at begin, 4: remove token at end*/
    int operatorType = curand(state) % numOperators;

    if (parent->tokenSize >= MAX_RULE_SIZE && operatorType < 3 && operatorType != 0)
    {
        operatorType += 2;
    }

    // copy parent to child when operation is 0, 2, or 4
    if (operatorType % 2 == 0)
    {
        for (int i = 0; i < parent->tokenSize * 3; i++)
        {
            child->tokens[i] = parent->tokens[i];
        }
    }

    if (operatorType == 0)
    {
        // mutatation
        // Select a random chromosone
        int chromosoneId = curand(state) % parent->tokenSize;
        // Insert the feature in the chromosone
        createToken(child, chromosoneId, featureSize, state);
    }
    else if (operatorType == 1)
    {
        // add token at begin
        // move all token to right
        for (int i = 0; i < parent->tokenSize * 3; i++)
        {
            child->tokens[i + 3] = parent->tokens[i];
        }
        // create new token
        createToken(child, 0, featureSize, state);
        child->tokenSize++;
    }
    else if (operatorType == 2)
    {
        // add token at end
        // create new token
        createToken(child, parent->tokenSize, featureSize, state);
        child->tokenSize++;
    }
    else if (operatorType == 3)
    {
        // remove token at begin
        // move all token to left
        for (int i = 0; i < parent->tokenSize * 3 - 3; i++)
        {
            child->tokens[i] = parent->tokens[i + 3];
        }
        child->tokenSize--;
    }
    else if (operatorType == 4)
    {
        // remove token at end
        child->tokenSize--;
    }

    if (child->tokenSize == 1 && child->tokens[2] == 0)
    {
        child->tokens[2] = 1;
    }
    child->fitness = 0;
}

__global__ void
kernelInitRules(
    population_t *init_population,
    int n, int nDocs, int initDisplacement,
    int totalThreads,
    document_t *doc_list,
    int *featureSize,
    int *tagArray,
    curandState_t *states)
{
    int i, j;
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

        initIteration += initDisplacement;
        endIteration += initDisplacement;
        for (i = initIteration; i < endIteration; i++)
        {
            int ruleSize = 1; //(curand(&states[i]) % MAX_INIT_RULE_SIZE) + 1;
            // Create a rule
            // printf("tag id: %d %d/%d\n", (int)i / init_population->width, (int)i, init_population->width);
            init_population->rules[i].entityId = tagArray[(int)i / init_population->width];
            init_population->rules[i].tokenSize = ruleSize;
            // printf("%d rule.entityId %d, rule.tokenSize %d\n", i, init_population->rules[i].entityId, init_population->rules[i].tokenSize);
            for (j = 0; j < ruleSize; j++)
            {
                int featureLingKey = curand(&states[i]) % POS_LIST_SIZE;
                int featureLingValue = curand(&states[i]) % featureSize[featureLingKey];
                // int featureLingValue = 2;
                int includeEntity = 1; // curand(&states[i]) % 2;
                init_population->rules[i].tokens[j * 3] = featureLingKey;
                init_population->rules[i].tokens[j * 3 + 1] = featureLingValue;
                init_population->rules[i].tokens[j * 3 + 2] = includeEntity;
                // printf("%d %d %d\n", featureLingKey, featureLingValue, includeEntity);
                init_population->rules[i].fitness = 0;
                // init_population->rules[i].fitness = fitness(init_population->rules[i], doc_list, nDocs);
            }
            __syncthreads();
        }
        __syncthreads();
        // replace existing rules with new ones
        for (i = 0; i < totalThreads * 0.5; i++)
        {
            for (int j = initIteration; j < endIteration; j++)
            {
                int jDisplacement = j + initDisplacement;
                int nextDisplacement = jDisplacement + i + 1;
                // Sort offspring rules
                if (nextDisplacement % 2 == 0 && nextDisplacement < totalThreads)
                {
                    if (init_population->rules[jDisplacement] == init_population->rules[nextDisplacement])
                    {
                        rule_t mutatedRule;
                        mutation(&init_population->rules[nextDisplacement], &mutatedRule, featureSize, &states[index]);
                        mutatedRule.fitness = 0;
                        // mutatedRule.fitness = fitness(mutatedRule, doc_list, nDocs);
                        init_population->rules[nextDisplacement] = mutatedRule;
                    }
                }
                __syncthreads();
                if (nextDisplacement % 2 == 1 && nextDisplacement < totalThreads)
                {
                    if (init_population->rules[jDisplacement] == init_population->rules[nextDisplacement])
                    {
                        rule_t mutatedRule;
                        mutation(&init_population->rules[nextDisplacement], &mutatedRule, featureSize, &states[index]);
                        mutatedRule.fitness = 0;
                        // mutatedRule.fitness = fitness(mutatedRule, doc_list, nDocs);
                        init_population->rules[nextDisplacement] = mutatedRule;
                    }
                }
                __syncthreads();
            }
            __syncthreads();
        }
        __syncthreads();
        // re-calculate fitness
        for (i = initIteration; i < endIteration; i++)
        {
            int iDisplacement = i + initDisplacement;
            init_population->rules[iDisplacement].fitness = 0;
            init_population->rules[iDisplacement].fitness = fitness(init_population->rules[iDisplacement], doc_list, nDocs);
        }
        // sort rules
        __syncthreads();
        for (i = 0; i < n / 2; i++)
        {
            for (int j = initIteration; j < endIteration; j++)
            {
                int jDisplacement = j + initDisplacement;
                if (j % 2 == 0 && j < n - 1)
                {
                    if (init_population->rules[jDisplacement] > init_population->rules[jDisplacement + 1])
                    {
                        rule_t temp = init_population->rules[jDisplacement];
                        init_population->rules[jDisplacement] = init_population->rules[jDisplacement + 1];
                        init_population->rules[jDisplacement + 1] = temp;
                    }
                }
                __syncthreads();
                if (j % 2 == 1 && j < n - 1)
                {
                    if (init_population->rules[jDisplacement] > init_population->rules[jDisplacement + 1])
                    {
                        rule_t temp = init_population->rules[jDisplacement];
                        init_population->rules[jDisplacement] = init_population->rules[jDisplacement + 1];
                        init_population->rules[jDisplacement + 1] = temp;
                    }
                }
                __syncthreads();
            }
            __syncthreads();
        }
    }
}

/*
 * Print error message from CUDA
 */
void printError(cudaError_t err, string prevMsg = "")
{
    if (err != cudaSuccess)
    {
        printf("kernel %s failed with error %s\n", prevMsg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**Generate init population of rules based in list of Documents**/
map<string, vector<Individual>> generate_init_population(
    document_t *d_doc_list, population_t *&d_population,
    int &sizeTag,
    int *d_featureSize,
    map<int, int> countEntities,
    curandState_t *states, cudaStream_t *streamList,
    int num_doc,
    int gridSize,
    int treadsPerBlock)
{

    map<string, vector<Rule>> init_population;
    // Configure population to GPU
    int h_tagArray[sizeTag];
    int i = 0;
    for (auto &it : countEntities)
    {
        h_tagArray[i] = it.first;
        i++;
    }
    int *d_tagArray;
    checkCudaErrors(cudaMallocManaged(&d_tagArray, sizeof(int) * sizeTag));
    checkCudaErrors(cudaMemcpy(d_tagArray, h_tagArray, sizeof(int) * sizeTag, cudaMemcpyHostToDevice));

    int initTotalRules = INIT_POPULATION_SIZE * sizeTag;
    // Initialize random states

    checkCudaErrors(cudaMallocManaged(&d_population, sizeof(population_t)));
    d_population->depth = sizeTag;
    d_population->width = INIT_POPULATION_SIZE;
    checkCudaErrors(cudaMallocManaged(&(d_population->ruleSize), sizeof(int) * sizeTag));
    checkCudaErrors(cudaMallocManaged(&(d_population->rules), sizeof(rule_t) * initTotalRules));

    for (int i = 0; i < sizeTag; i++)
    {
        kernelInitRules<<<gridSize, treadsPerBlock, 0, streamList[i % 2]>>>(
            d_population,
            INIT_POPULATION_SIZE, num_doc,
            i * INIT_POPULATION_SIZE,
            gridSize * treadsPerBlock,
            d_doc_list,
            d_featureSize,
            d_tagArray,
            states);
        printf("Init tag: %d\n", i);
        checkCudaErrors(cudaStreamSynchronize(streamList[i % 2]));
    }
    map<string, vector<Individual>> individual_population;
    init_population.clear();
    return individual_population;
}

/*Save Rule in file*/
void saveRules(vector<Rule> &rules, string filename)
{
    ofstream file;
    file.open(filename);
    for (int i = 0; i < rules.size(); i++)
    {
        file << rules[i].toString() << endl;
    }
    file.close();
}

/**
 * @brief crossover
 * @param parent1
 * @param parent2
 * @param child1
 * @param child2
 * @param state
 */
__device__ void crossover(
    rule_t *parent1, rule_t *parent2,
    rule_t *child1, rule_t *child2,
    curandState_t *state)
{

    int cutoff = curand(state) % parent1->tokenSize;
    child1->entityId = parent1->entityId;
    child1->fitness = 0;
    child2->entityId = parent2->entityId;
    child2->fitness = 0;
    child1->tokenSize = 0;
    child2->tokenSize = 0;

    for (int i = 0; i < cutoff; i++)
    {
        if (i < parent1->tokenSize)
        {
            child1->tokens[i * 3] = parent1->tokens[i * 3];
            child1->tokens[i * 3 + 1] = parent1->tokens[i * 3 + 1];
            child1->tokens[i * 3 + 2] = parent1->tokens[i * 3 + 2];
            child1->tokenSize++;
        }
        if (i < parent2->tokenSize)
        {
            child2->tokens[i * 3] = parent2->tokens[i * 3];
            child2->tokens[i * 3 + 1] = parent2->tokens[i * 3 + 1];
            child2->tokens[i * 3 + 2] = parent2->tokens[i * 3 + 2];
            child2->tokenSize++;
        }
    }

    for (int i = cutoff; i < parent1->tokenSize; i++)
    {
        if (i < parent2->tokenSize)
        {
            child1->tokens[i * 3] = parent2->tokens[i * 3];
            child1->tokens[i * 3 + 1] = parent2->tokens[i * 3 + 1];
            child1->tokens[i * 3 + 2] = parent2->tokens[i * 3 + 2];
            child1->tokenSize++;
        }
        if (i < parent1->tokenSize)
        {
            child2->tokens[i * 3] = parent1->tokens[i * 3];
            child2->tokens[i * 3 + 1] = parent1->tokens[i * 3 + 1];
            child2->tokens[i * 3 + 2] = parent1->tokens[i * 3 + 2];
            child2->tokenSize++;
        }
    }

    if (child1->tokenSize == 1 && child1->tokens[2] == 0)
    {
        child1->tokens[2] = 1;
    }
}

/**
 * @brief train population according a tagname
 */
__global__ void train_population(
    volatile population_t *population,
    volatile population_t *bestPopulation,
    volatile population_t *offspring,
    int initDisplacement,
    int *featureSize,
    volatile float *bestFitness,
    volatile int *iterations,
    curandState_t *states,
    int nDocs, document_t *doc_list,
    int totalThreads,
    featureOccurrenceCollection_t *d_featureOccurrenceCollection,
    int tol = 7, int max_iter = 1000,
    float best_pecent = 0.1,
    bool simple = true)
{

    __shared__ volatile bool tagFinished;
    extern __shared__ volatile int s_iterations[];
    extern __shared__ volatile float s_bestFitness[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int localIndex = threadIdx.x;
    int indexRule = (int)initDisplacement / population->width;
    int initIteration, endIteration;
    int i;
    int totalRules = population->width;

    if (index == 0)
    {
        printf("totalRules: %d,  indexRule: %d\n", totalRules, indexRule);
        printf("size features: %d\n", d_featureOccurrenceCollection->size);
    }

    if (index < totalRules)
    {
        // Init control variables
        if (localIndex == 0)
        {
            tagFinished = false;
            s_iterations[0] = iterations[indexRule * 2];
            s_iterations[1] = iterations[indexRule * 2 + 1];
            s_bestFitness[0] = bestFitness[indexRule * 4];
            s_bestFitness[1] = bestFitness[indexRule * 4 + 1];
            s_bestFitness[2] = bestFitness[indexRule * 4 + 2];
            s_bestFitness[3] = bestFitness[indexRule * 4 + 3];
        }
        __syncthreads();

        // Train loop
        while (!tagFinished)
        {
            __syncthreads();
            // config displacement
            if (totalRules < totalThreads)
            {
                initIteration = index;
                endIteration = index + 1;
            }
            else
            {
                float increment = (float)totalRules / totalThreads;
                initIteration = increment * (index);
                endIteration = increment * (index + 1);
            }
            // wait for all threads for access to displacements

            // create new individuals
            __syncthreads();
            for (i = initIteration; i < endIteration; i++)
            {
                int displacementIndex = i + initDisplacement;
                if (displacementIndex < totalRules)
                {
                    int currentIteration = iterations[indexRule * 2];
                    float mutation_rate = exp(-currentIteration / max_iter);
                    float crossover_rate = (1 - mutation_rate);
                    // printf("displacementIndex %d i %d\n", displacementIndex, i);
                    rule_t parent = population->rules[displacementIndex];
                    // select genetic operator, mutation 0 or crossover 1

                    int geneticOperator = (parent.tokenSize > 1 ? (curand(&states[i]) % 100 < mutation_rate * 100 ? 0 : 1) : 0);
                    // if mutation
                    rule_t child;

                    if (geneticOperator == 0)
                    {
                        mutation(&parent, &child, featureSize, &states[i]);
                    }
                    else
                    {
                        // if crossover
                        int parent2Index = (indexRule * population->width) + (curand(&states[i]) % population->width);
                        rule_t parent2 = population->rules[parent2Index];
                        rule_t child2;
                        crossover(&parent, &parent2, &child, &child2, &states[i]);
                        child2.fitness = 0.0;
                    }
                    child.fitness = 0.0;
                    // child.fitness = fitness(child, doc_list, nDocs);
                    // printf("child fitness %f\n", child.fitness);
                    offspring->rules[displacementIndex] = child;
                }
                __syncthreads();
            }

            // survival selection
            __syncthreads();
            // replace existing rules with new ones
            for (i = 0; i < totalThreads / 2; i++)
            {
                for (int j = initIteration; j < endIteration; j++)
                {
                    int jDisplacement = j + initDisplacement;
                    int nextDisplacement = jDisplacement + i + 1;
                    // Sort offspring rules
                    if (nextDisplacement % 2 == 0 && nextDisplacement < totalThreads)
                    {
                        if (population->rules[jDisplacement] == population->rules[nextDisplacement])
                        {
                            rule_t mutatedRule;
                            mutation(&population->rules[nextDisplacement], &mutatedRule, featureSize, &states[index]);
                            mutatedRule.fitness = 0.0;
                            // mutatedRule.fitness = fitness(mutatedRule, doc_list, nDocs);
                            population->rules[nextDisplacement] = mutatedRule;
                        }
                    }
                    __syncthreads();
                    if (nextDisplacement % 2 == 1 && nextDisplacement < totalThreads)
                    {
                        if (population->rules[jDisplacement] == population->rules[nextDisplacement])
                        {
                            rule_t mutatedRule;
                            mutation(&population->rules[nextDisplacement], &mutatedRule, featureSize, &states[index]);
                            mutatedRule.fitness = 0.0;
                            // mutatedRule.fitness = fitness(mutatedRule, doc_list, nDocs);
                            population->rules[nextDisplacement] = mutatedRule;
                        }
                    }
                    __syncthreads();
                }
                __syncthreads();
            }
            // re-calculate fitness
            for (i = initIteration; i < endIteration; i++)
            {
                int displacementIndex = i + initDisplacement;
                if (displacementIndex < totalRules)
                {
                    rule_t rule = population->rules[displacementIndex];
                    rule.fitness = 0.0;
                    rule.fitness = fitness(rule, doc_list, nDocs);
                    population->rules[displacementIndex] = rule;

                    rule_t rule2 = offspring->rules[displacementIndex];
                    rule2.fitness = 0.0;
                    rule2.fitness = fitness(rule2, doc_list, nDocs);
                    offspring->rules[displacementIndex] = rule2;
                }
                __syncthreads();
            }
            // sort rules
            __syncthreads();
            for (i = 0; i < totalThreads / 2; i++)
            {
                for (int j = initIteration; j < endIteration; j++)
                {
                    int jDisplacement = j + initDisplacement;
                    // Sort offspring rules
                    if (j % 2 == 0 && j < totalThreads - 1)
                    {
                        if (offspring->rules[jDisplacement] < offspring->rules[jDisplacement + 1])
                        {
                            rule_t temp = offspring->rules[jDisplacement];
                            offspring->rules[jDisplacement] = offspring->rules[jDisplacement + 1];
                            offspring->rules[jDisplacement + 1] = temp;
                        }
                    }
                    __syncthreads();
                    if (j % 2 == 1 && j < totalThreads - 1)
                    {
                        if (offspring->rules[jDisplacement] < offspring->rules[jDisplacement + 1])
                        {
                            rule_t temp = offspring->rules[jDisplacement];
                            offspring->rules[jDisplacement] = offspring->rules[jDisplacement + 1];
                            offspring->rules[jDisplacement + 1] = temp;
                        }
                    }
                    __syncthreads();
                    // change the parent rules to the offspring rules
                    if (j < totalThreads)
                    {
                        if (population->rules[jDisplacement] < offspring->rules[jDisplacement])
                        {
                            rule_t temp = population->rules[jDisplacement];
                            population->rules[jDisplacement] = offspring->rules[jDisplacement];
                            offspring->rules[jDisplacement] = temp;
                        }
                    }
                    __syncthreads();
                }
                __syncthreads();
            }

            // Search for best fitness by tag
            __syncthreads();
            if (localIndex == 0 && !tagFinished)
            {
                s_iterations[0] = iterations[indexRule * 2];
                s_iterations[1] = iterations[indexRule * 2 + 1];

                // printf("index %d: tag finished:%d\n", index, tagFinished[index]);
                // printf("Class %d: %d/%d\n", index, iterations[index * 2], max_iter);
                initIteration = initDisplacement;
                endIteration = initDisplacement + population->width;
                float bestCurrentFitness = 0.0;
                float avgCurrentFitness = 0.0;
                // get best fitness
                for (i = initIteration; i < endIteration; i++)
                {
                    avgCurrentFitness += population->rules[i].fitness;
                    if (population->rules[i].fitness > bestCurrentFitness)
                    {
                        bestCurrentFitness = population->rules[i].fitness;
                    }
                }
                // printf("bestCurrentFitness %f\n", bestCurrentFitness);

                avgCurrentFitness /= population->width;
                bool improve = false;

                if (bestCurrentFitness - (1.0 / max_iter) > s_bestFitness[0])
                {
                    improve = true;
                    s_bestFitness[0] = bestCurrentFitness;
                }
                if (avgCurrentFitness - (1 / max_iter) > s_bestFitness[2])
                {
                    s_bestFitness[2] = avgCurrentFitness;
                    improve = true;
                }
                s_iterations[0] += 1;
                iterations[indexRule * 2] += 1;
                if (improve)
                {
                    s_iterations[1] = 0;
                    iterations[indexRule * 2 + 1] = 0;
                    // save best population by tag
                    for (int j = initDisplacement; j < initDisplacement + population->width; j++)
                    {
                        bestPopulation->rules[j] = population->rules[j];
                    }
                    // printf("Tag %d: %d/%d improve %d\n", indexRule, s_iterations[0], max_iter, index);
                }
                else
                {
                    s_iterations[1] += 1;
                    iterations[indexRule * 2 + 1] += 1;
                }
                // printf("current tol %d: %d > tol:%d\n", localIndex, s_iterations[1], tol);
            }

            // Evaluate tolerance and max iterations by tag
            __syncthreads();
            if (index == 0)
            {
                iterations[indexRule * 2] += 1;
            }
            __syncthreads();
            if (localIndex == 0)
            {
                s_iterations[0] = iterations[indexRule * 2];
                iterations[indexRule * 2 + 1] = min(abs(s_iterations[1]), abs(iterations[indexRule * 2 + 1]));
                bestFitness[indexRule * 4] = min(abs(s_bestFitness[0]), abs(bestFitness[indexRule * 4]));
                bestFitness[indexRule * 4 + 2] = min(abs(s_bestFitness[2]), abs(bestFitness[indexRule * 4 + 2]));

                if (abs(iterations[indexRule * 2 + 1]) > tol || abs(iterations[indexRule * 2]) > max_iter)
                {
                    tagFinished = true;
                }
                __syncthreads();
                s_iterations[1] = iterations[indexRule * 2 + 1];
                s_bestFitness[0] = bestFitness[indexRule * 4];
                s_bestFitness[2] = bestFitness[indexRule * 4 + 2];
            }

            // Save best fitness
            __syncthreads();
            if (localIndex == 0 && tagFinished)
            {
                // printf("index %d: tag finished:%d\n", index, tagFinished[index]);
                // printf("Class %d: %d/%d\n", index, iterations[index * 2], max_iter);
                initIteration = initDisplacement;
                endIteration = initDisplacement + population->width;
                float bestCurrentFitness = 0.0;
                float avgCurrentFitness = 0.0;
                // get best fitness
                for (i = initIteration; i < endIteration; i++)
                {
                    avgCurrentFitness += bestPopulation->rules[i].fitness;
                    if (bestPopulation->rules[i].fitness > bestCurrentFitness)
                    {
                        bestCurrentFitness = bestPopulation->rules[i].fitness;
                    }
                }
                avgCurrentFitness /= bestPopulation->width;
                bestFitness[indexRule * 4] = bestCurrentFitness;
                bestFitness[indexRule * 4 + 1] = s_bestFitness[1];
                bestFitness[indexRule * 4 + 2] = avgCurrentFitness;
                bestFitness[indexRule * 4 + 3] = s_bestFitness[3];
                // iterations[localIndex * 2] = s_iterations[localIndex * 2];
                // iterations[localIndex * 2 + 1] = s_iterations[localIndex * 2 + 1];
            }
            __syncthreads();
        }
    }
    // __syncthreads();
    // if (index == 0)
    // {
    //     for (int i = 0; i < population->width; i++)
    //     {
    //         if (population->rules[i].fitness > 0)
    //         {
    //             for (int j = 0; j < population->rules[i].tokenSize * POS_LIST_SIZE; j++)
    //             {
    //                 printf("%d ", population->rules[i].tokens[j]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }
}

/**
 * @brief Split data into training and test
 */
void splitData(
    vector<Document *> &population,
    vector<Document *> &training_population,
    vector<Document *> &test_population,
    float train_percent = 0.8)
{

    int train_size = (int)(population.size() * train_percent);
    float theshold = train_size < 10 ? train_size : train_percent * 10;
    for (int i = 0; i < population.size(); i++)
    {
        if (i % 10 < theshold)
        {
            training_population.push_back(population[i]);
        }
        else
        {
            test_population.push_back(population[i]);
        }
    }
}

void splitData(
    int *&training_population,
    int *&test_population,
    int n,
    float train_percent = 0.8)
{

    int train_size = (int)(n * train_percent);
    float theshold = train_size < 10 ? train_size : train_percent * 10;
    training_population = (int *)malloc(sizeof(int) * train_size);
    test_population = (int *)malloc(sizeof(int) * (n - train_size));
    int j = 0;
    int k = 0;
    for (int i = 0; i < n; i++)
    {
        if (i % 10 < theshold)
        {
            training_population[j] = i;
            j++;
        }
        else
        {
            training_population[k] = i;
            k++;
        }
    }
}

void saveRulesFromCuda(
    population_t *population,
    int tagId, int totalThreads,
    map<int, string> id2tagSimple,
    map<string, map<int, string>> vocab_map, document_t document)
{
    vector<Rule> bestRules;
    int initIteration = tagId * population->width;
    int endIteration = (tagId + 1) * population->width;

    printf("Tag %d-%s: %d/%d\n", tagId, id2tagSimple[tagId].c_str(), initIteration, endIteration);
    for (int i = initIteration; i < endIteration; i++)
    {
        if (population->rules[i].fitness > 0)
        {
            printf("Rule %d: %f\n", i, population->rules[i].fitness);
            for (int j = 0; j < population->rules[i].tokenSize; j++)
            {
                printf("%d %d %d, ", population->rules[i].tokens[j * 3], population->rules[i].tokens[j * 3 + 1], population->rules[i].tokens[j * 3 + 2]);
            }
            printf("\n");
            int spanList[360];
            int spanSize = get_span_list(population->rules[i], document, spanList, document.tokenSize / 3);
            printf("spanSize:%d\n", spanSize);
            printf("fitness: %f\n", fitness(population->rules[i], &document, 1));
            Rule gaRule(&population->rules[i], id2tagSimple, vocab_map);
            if (!gaRule.isIn(bestRules))
                bestRules.push_back(gaRule);
        }
    }
    // Sort rules by score
    if (bestRules.size() > 0)
    {
        sort(bestRules.begin(), bestRules.end(), std::greater<>());
        string tag = bestRules[0].tag;
        saveRules(bestRules, "./ga_rules/" + tag + "_" + to_string(totalThreads) + ".txt");
    }
    bestRules.clear();
}

#include <cuda.h>
#include <cuda_runtime.h>

feature_t createFeature(
    int token,
    int tokenPosition,
    unsigned int *featureStartPositions,
    map<string, map<int, string>> vocab_map,
    map<string, map<string, int>> vocab2id_map)
{
    unsigned short typeId = tokenPosition % POS_LIST_SIZE;
    string feature = POS_LIST[typeId];
    string feaure_value = vocab_map[feature][token];

    int feature_value_id = vocab2id_map[feature][feaure_value];

    unsigned int feature_id = featureStartPositions[typeId] + feature_value_id;
    // printf("%s %s %d\n", feature.c_str(), feaure_value.c_str(), feature_value_id);

    return feature_t(feature_id, feature_value_id, typeId);
}

int main(int argc, char *argv[])
{

    // Define variables
    int nDocs = 100;
    int tol = 7;
    int maxItr = 100;

    document_t *document_pnt;
    int *featureSize;
    int *test_document_pnt;
    int *train_document_pnt;
    int sizeTag;
    map<int, string> id2tagSimple;
    map<string, map<int, string>> vocab_map;
    map<string, map<string, int>> vocab2id_map;

    // Read arguments
    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "-nDocs") == 0)
        {
            nDocs = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-tol") == 0)
        {
            tol = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-maxItr") == 0)
        {
            maxItr = atoi(argv[i + 1]);
            i++;
        }
    }
    vector<Document *>
        doc_vector = read_data(
            "/mnt/d/data/HDF5_FILE.h5",
            document_pnt, featureSize,
            id2tagSimple,
            vocab_map,
            vocab2id_map);
    vector<Document *> training_population;
    vector<Document *> test_population;
    printf("%d\n", doc_vector.size());
    srand(42);
    splitData(doc_vector, training_population, test_population);
    splitData(test_document_pnt, train_document_pnt, 2111);
    int *d_featureSize;
    checkCudaErrors(cudaMalloc(&d_featureSize, sizeof(int) * POS_LIST_SIZE));
    checkCudaErrors(cudaMemcpy(d_featureSize, featureSize, sizeof(int) * POS_LIST_SIZE, cudaMemcpyHostToDevice));

    int indexThreadList[] = {8};
    int numThreads = sizeof(indexThreadList) / sizeof(int);
    ofstream timeFile;
    timeFile.open("./ga_result/time.txt");

    ofstream timeFileInit;
    timeFileInit.open("./ga_result/time_init.txt");

    // Initialize where each feature is started to global id
    unsigned int featureStartPositions[POS_LIST_SIZE];
    featureStartPositions[0] = 0;
    for (int i = 1; i < POS_LIST_SIZE; i++)
    {
        featureStartPositions[i] = featureStartPositions[i - 1] + featureSize[i - 1];
    }

    for (int N = 4; N <= 4; N++)
    {
        for (int x = 0; x < numThreads; x++)
        {
            struct timeval tval_beforei;
            struct timeval tval_before, tval_after, tval_result;

            int threadsPerblock = (128 / N) * indexThreadList[x];
            int num_threads = threadsPerblock * N;
            printf("Num threads:%d\n", num_threads);

            cudaStream_t streamList[2];
            for (int j = 0; j < 2; j++)
            {
                cudaStreamCreate(&streamList[j]);
            }

            gettimeofday(&tval_beforei, NULL);
            curandState_t *states;
            population_t *d_population;
            population_t *d_bestPopulation;
            population_t *d_offspintPopulation;

            document_t *d_doc_list;
            // Copy document data to GPU
            checkCudaErrors(cudaMallocManaged(&d_doc_list, sizeof(document_t) * nDocs));
            map<int, int> countEntities;

            // Copy nDocs documents data to GPU and create transtion features
            for (int i = 0; i < nDocs; i++)
            {
                int *tokens = (int *)malloc(sizeof(int) * document_pnt[i].tokenSize);
                checkCudaErrors(cudaMallocManaged(&tokens, sizeof(int) * document_pnt[i].tokenSize));
                checkCudaErrors(cudaMemcpy(&d_doc_list[i], &document_pnt[i], sizeof(document_t), cudaMemcpyHostToDevice));

                for (int j = 0; j < document_pnt[i].tokenSize; j++)
                {
                    tokens[j] = document_pnt[i].tokens[j];
                }
                d_doc_list[i].tokens = tokens;

                int *entities = (int *)malloc(sizeof(int) * document_pnt[i].entitySize);
                checkCudaErrors(cudaMallocManaged(&entities, sizeof(int) * document_pnt[i].entitySize));
                for (int j = 0; j < document_pnt[i].entitySize; j++)
                {
                    entities[j] = document_pnt[i].entities[j];

                    if (j % 3 == 0)
                    {
                        countEntities[entities[j]]++;
                    }
                }
                d_doc_list[i].entities = entities;
            };

            // Fii feature occurrence for each feature
            featureOccurrenceCollection_t featureOccurrenceCollection;
            for (int i = 0; i < nDocs; i++)
            {
                for (int j = 0; j < document_pnt[i].tokenSize; j++)
                {
                    // Map occurence for each feature
                    for (int k = 0; k < POS_LIST_SIZE; k++)
                    {
                        feature_t currentFeature = createFeature(document_pnt[i].tokens[j], j, featureStartPositions, vocab_map, vocab2id_map);
                        featureOccurrence_t featureOccurrence = featureOccurrenceCollection.get_feature(currentFeature.id);
                        printf("id: %d, value:%d, type:%d\n", currentFeature.id, currentFeature.value, currentFeature.type);
                        feature_t prevFeature;
                        int prevIndex = (int)(j / POS_LIST_SIZE - 1) * POS_LIST_SIZE;
                        if (prevIndex < 1)
                        {
                            prevFeature = feature_t(0, 0, 0);
                        }
                        else
                        {
                            prevIndex += k;
                            prevFeature = createFeature(document_pnt[i].tokens[prevIndex], prevIndex, featureStartPositions, vocab_map, vocab2id_map);
                        }

                        int nextIndex = (int)(j / POS_LIST_SIZE + 1) * POS_LIST_SIZE;
                        feature_t nextFeature;
                        if (nextIndex >= document_pnt[i].tokenSize)
                        {
                            nextFeature = feature_t(0, 0, 0);
                        }
                        else
                        {
                            nextIndex += k;
                            nextFeature = createFeature(document_pnt[i].tokens[nextIndex], nextIndex, featureStartPositions, vocab_map, vocab2id_map);
                        }

                        if (featureOccurrence.feature.id == -1)
                        {
                            featureOccurrence = featureOccurrence_t(currentFeature);
                        }
                        printf("prev:%d, next:%d\n", prevIndex, nextIndex);
                        featureOccurrence.addTransition(prevFeature, nextFeature);
                        featureOccurrenceCollection.addFeature(featureOccurrence);
                    }
                }
            }

            featureOccurrenceCollection.calculateProbability();
            featureOccurrenceCollection.print();
            // Copy feature occurrence collection to GPU
            featureOccurrenceCollection_t *d_featureOccurrenceCollection;
            checkCudaErrors(cudaMallocManaged(&d_featureOccurrenceCollection, sizeof(featureOccurrenceCollection_t)));
            for (int i = 0; i < featureOccurrenceCollection.size(); i++)
            {
                featureOccurrence_t featureOccurrence = featureOccurrenceCollection.featureOccurences[i];
                featureOccurrence_t *d_featureOccurrence;
                checkCudaErrors(cudaMallocManaged(&d_featureOccurrence, sizeof(featureOccurrence_t) * featureOccurrence.size()));
                for (int j = 0; j < featureOccurrence.size(); j++)
                {
                    checkCudaErrors(cudaMallocManaged(&d_featureOccurrence, sizeof(featureOccurrence_t) * featureOccurrence.size()));
                }
                        }

            sizeTag = countEntities.size();
            int initTotalRules = INIT_POPULATION_SIZE * sizeTag;
            cudaMalloc((void **)&states, initTotalRules * sizeof(curandState_t));
            init<<<N, threadsPerblock, 0, streamList[0]>>>(42, initTotalRules, num_threads, states);
            checkCudaErrors(cudaDeviceSynchronize());

            map<string, vector<Individual>>
                population = generate_init_population(
                    d_doc_list, d_population, sizeTag, d_featureSize,
                    countEntities,
                    states, streamList, nDocs,
                    N, threadsPerblock);
            printf("Size tag: %d\n", sizeTag);

            gettimeofday(&tval_after, NULL);
            timersub(&tval_after, &tval_beforei, &tval_result);
            timeFileInit << N << "," << threadsPerblock << "," << num_threads << "," << tval_result.tv_sec << "." << tval_result.tv_usec << endl;

            gettimeofday(&tval_before, NULL);
            // INI BEST FITNESS
            // first position is the best fitness, second position is previous best fitness,
            // third position is average fitness, fourth
            float *d_bestFitness = (float *)malloc(sizeof(float) * sizeTag * 4);
            checkCudaErrors(
                cudaMallocManaged(&d_bestFitness, sizeof(float) * sizeTag * 4));
            for (int i = 0; i < sizeTag * 4; i++)
            {
                d_bestFitness[i] = -1.0;
            }
            // INIT iterations
            int *h_iterations = (int *)malloc(sizeof(int) * sizeTag * 2);
            int *d_iterations;
            checkCudaErrors(cudaMalloc(&d_iterations, sizeof(int) * sizeTag * 2));
            for (int i = 0; i < sizeTag * 2; i++)
            {
                h_iterations[i] = 0;
            }
            checkCudaErrors(cudaMemcpy(d_iterations, h_iterations, sizeof(int) * sizeTag, cudaMemcpyHostToDevice));
            printf("Init population\n");

            // Initialize best population
            checkCudaErrors(cudaMallocManaged(&d_bestPopulation, sizeof(population_t)));
            d_bestPopulation->depth = sizeTag;
            d_bestPopulation->width = INIT_POPULATION_SIZE;
            checkCudaErrors(cudaMallocManaged(&(d_bestPopulation->ruleSize), sizeof(int) * sizeTag));
            checkCudaErrors(cudaMallocManaged(&(d_bestPopulation->rules), sizeof(rule_t) * initTotalRules));

            checkCudaErrors(cudaMallocManaged(&d_offspintPopulation, sizeof(population_t)));
            d_offspintPopulation->depth = sizeTag;
            d_offspintPopulation->width = INIT_POPULATION_SIZE;
            checkCudaErrors(cudaMallocManaged(&(d_offspintPopulation->ruleSize), sizeof(int) * sizeTag));
            checkCudaErrors(cudaMallocManaged(&(d_offspintPopulation->rules), sizeof(rule_t) * initTotalRules));

            checkCudaErrors(cudaDeviceSynchronize());
            for (int i = 0; i < sizeTag; i++)
            {
                train_population<<<N, threadsPerblock, 0, streamList[i % 2]>>>(
                    d_population, d_bestPopulation, d_offspintPopulation,
                    i * INIT_POPULATION_SIZE,
                    d_featureSize,
                    d_bestFitness,
                    d_iterations,
                    states,
                    nDocs,
                    d_doc_list,
                    num_threads,
                    d_featureOccurrenceCollection,
                    tol,
                    maxItr, 0.1, false);
                printf("Training population %d %dX%d=%d\n", i, N, threadsPerblock, num_threads);
                checkCudaErrors(cudaDeviceSynchronize());
                // checkCudaErrors(cudaStreamSynchronize(streamList[i % 2]));
            }

            checkCudaErrors(cudaMemcpy(h_iterations, d_iterations, sizeof(int) * sizeTag * 2, cudaMemcpyDeviceToHost));

            // Print Document
            document_t doc = d_doc_list[0];
            printf("Tokens: \n");
            for (int i = 0; i < d_doc_list[0].tokenSize; i++)
            {
                printf("%d ", d_doc_list[0].tokens[i]);
                if (i % 3 == 2)
                    printf("\n");
            }
            printf("\n");
            cudaFree(d_iterations);
            cudaFree(states);
            cudaFree(d_bestFitness);
            cudaFree(d_doc_list);

            cudaFree(d_offspintPopulation);

            // Save cuda rules
            for (int i = 0; i < sizeTag; i++)
            {
                saveRulesFromCuda(
                    d_bestPopulation, i, num_threads,
                    id2tagSimple, vocab_map, doc);
            }
            cudaFree(d_population);
            cudaFree(d_bestPopulation);
            // print iteration data
            int total_itr = 0;
            checkCudaErrors(cudaDeviceSynchronize());
            for (int i = 0; i < sizeTag; i++)
            {
                printf("Iteration %d: %d \n", i, h_iterations[i * 2]);
                total_itr += h_iterations[i * 2];
            }

            // free memory
            for (int i = 0; i < 2; i++)
            {
                checkCudaErrors(cudaStreamDestroy(streamList[i]));
            }
            gettimeofday(&tval_after, NULL);
            timersub(&tval_after, &tval_before, &tval_result);
            timeFile << N << "," << threadsPerblock << "," << num_threads << "," << tval_result.tv_sec << "." << tval_result.tv_usec << "," << total_itr << endl;
            printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
        }
    }
    timeFileInit.close();
    timeFile.close();
    free(document_pnt);
    return 0;
}
