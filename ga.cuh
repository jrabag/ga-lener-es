#include <sstream>

int INIT_POPULATION_SIZE = 1000;
__device__ __managed__ int POS_LIST_SIZE = 3;
__device__ __managed__ int MAX_INIT_RULE_SIZE = 2;
__device__ __managed__ int MAX_RULE_SIZE = 11;

std::vector<std::string>
    POS_LIST{"dep_", "lemma_", "pos_"};

int *equalint = new int(1);
int *diffint = new int(0);
#include <cuda/std/cstddef>
#include <cuda/std/tuple>
#include <tuple>

#include <helper_functions.h>
#include <helper_cuda.h>
/**
 * Class document
 * Representation for a text using PoS tags
 */
class Document
{
public:
    __host__ Document(){};
    std::map<std::string, std::vector<std::tuple<int, int>>> entities;
    std::vector<std::map<std::string, std::string>> tokens;
    std::map<std::string, std::vector<std::tuple<int, int>>> getEntities() { return entities; }
    std::vector<std::map<std::string, std::string>> getTokens() { return tokens; }

    __host__ int size()
    {
        return tokens.size();
    }
    __host__ std::map<std::string, std::string> getToken(int index)
    {
        return tokens[index];
    }
    __host__ std::tuple<std::string, std::string> getTupleToken(int index)
    {
        std::map<std::string, std::string> token = getToken(index);
        return std::make_tuple(token.begin()->first, token.begin()->second);
    }

    __host__ std::tuple<std::string, std::string> getTupleToken(int index, std::string pos_key)
    {
        std::map<std::string, std::string> token = getToken(index);
        return std::make_tuple(pos_key, token[pos_key]);
    }

    __host__ std::vector<std::tuple<std::string, std::string>> getTuplesToken(int index)
    {
        std::map<std::string, std::string> token = getToken(index);
        std::vector<std::tuple<std::string, std::string>> token_tuple;
        for (auto it = token.begin(); it != token.end(); it++)
        {
            token_tuple.push_back(getTupleToken(index, it->first));
        }
        return token_tuple;
    }
};

/**
 * Class TokenRule
 * Representation for a PoS Rule Token
 */
class RuleToken
{
public:
    std::tuple<std::string, std::string> pos;
    bool is_include;
    int document_id;
    int token_position;
    __host__ RuleToken(std::tuple<std::string, std::string> pos, bool is_include, int document_id, int token_position);
    void _print();

    __host__ bool operator==(const RuleToken &other) const
    {
        return std::get<1>(pos) == std::get<1>(other.pos) && std::get<0>(pos) == std::get<0>(other.pos);
    }

    __host__ std::string toString()
    {
        std::stringstream ss;
        ss << "{\"" << std::get<0>(pos) << "\": \"" << std::get<1>(pos) << "\", \"include\":" << is_include;

        if (document_id >= 0)
        {
            ss << ", \"document_id\":" << document_id;
        }

        if (token_position >= 0)
        {
            ss << ", \"token_position\":" << token_position;
        }

        ss << "}";

        return ss.str();
    }
};

std::map<int, std::string> read_string(hid_t hfile, const char *dset_name);

struct document_t
{
    int tokenSize;
    int *tokens;
    int entitySize;
    int *entities;
    // free pointers
    void operator~()
    {
        free(tokens);
        free(entities);
    }
};

struct rule_t
{
    int entityId;
    int tokenSize;
    int tokens[3 * 20]; // Feature + Value + Include
    float fitness;

    __device__ __host__ rule_t operator=(const rule_t &other)
    {
        entityId = other.entityId;
        tokenSize = other.tokenSize;
        fitness = other.fitness;
        for (int i = 0; i < 20 * 3; i++)
        {
            tokens[i] = other.tokens[i];
        }
        return *this;
    }

    __device__ __host__ bool operator==(const rule_t &other) const
    {
        bool meta_equal = entityId == other.entityId && tokenSize == other.tokenSize;
        if (!meta_equal)
            return false;
        for (int i = 0; i < tokenSize; i++)
        {
            if (tokens[i * 3] != other.tokens[i * 3])
            {
                return false;
            }
            if (tokens[i * 3 + 1] != other.tokens[i * 3 + 1])
            {
                return false;
            }
            if (tokens[i * 3 + 2] != other.tokens[i * 3 + 2])
            {
                return false;
            }
        }
        return true;
    }

    // evalaute if rule greater than other rule
    __device__ __host__ bool operator>(const rule_t &other) const
    {
        if (fitness > other.fitness)
        {
            return true;
        }
        if (tokenSize > other.tokenSize)
        {
            return true;
        }
        return false;
    }

    __device__ __host__ bool operator<(const rule_t &other) const
    {
        if (fitness < other.fitness)
        {
            return true;
        }
        if (tokenSize < other.tokenSize)
        {
            return true;
        }
        return false;
    }
};

struct population_t
{
    int depth; // Number of rules by tag
    int width; // Number of different tags
    rule_t *rules;
    int *ruleSize;
    void operator~()
    {
        free(rules);
        free(ruleSize);
    }
    bool operator==(const population_t &other) const
    {
        return depth == other.depth && width == other.width;
    }
};

/**
 * Class Rule
 * Representation for a PoS rule
 */
class Rule
{

public:
    std::vector<RuleToken> token_list;
    std::string tag;
    float score;

    Rule();
    Rule(std::string tag);
    Rule(std::string tag, std::vector<RuleToken> token_list);
    Rule(
        rule_t *rule,
        std::map<int, std::string> id2tagSimple,
        std::map<std::string, std::map<int, std::string>> vocab_map)
    {
        this->tag = id2tagSimple[rule->entityId];
        this->score = rule->fitness;
        for (int i = 0; i < rule->tokenSize; i++)
        {
            std::string feature = POS_LIST[rule->tokens[i * 3]];
            std::string token_value = vocab_map[feature][rule->tokens[i * 3 + 1]];
            bool is_include = rule->tokens[i * 3 + 2] == 1;
            this->token_list.push_back(
                RuleToken(
                    std::make_tuple(feature, token_value),
                    is_include,
                    -1, -1));
        }
    }

    void add_token(RuleToken token);
    void add_token(std::tuple<std::string, std::string> pos, bool is_include, int document_id, int token_position);
    void insert_token(std::tuple<std::string, std::string> pos, bool is_include, int document_id, int token_position);
    float fitness(Document doc_list[], int docSize);
    float fitness(std::vector<Document *> doc_list);
    std::vector<std::tuple<int, int>> get_span_list(Document *doc);

    bool operator==(Rule &other) const
    {
        return tag == other.tag && token_list.size() == other.token_list.size() && token_list == other.token_list;
    }

    int size()
    {
        return token_list.size();
    }

    /*Eval if rule is in rule vector*/
    bool isIn(std::vector<Rule> rule)
    {
        for (auto it = rule.begin(); it != rule.end(); it++)
        {
            if (*it == *this)
            {
                return true;
            }
        }
        return false;
    }

    // evalaute if rule greater than other rule
    bool operator>(Rule &other) const
    {
        if (score > other.score)
        {
            return true;
        }
        return false;
    }

    std::string toString()
    {
        std::string rule_str = "{\"" + tag + "\" :[";
        for (auto it = token_list.begin(); it != token_list.end(); it++)
        {
            rule_str += it->toString() + (it != token_list.end() - 1 ? "," : "");
        }
        return rule_str + "]}";
    }
};

// Struct to represent a Feature in documents
struct feature_t
{
    long id;
    int value;
    short type;
    unsigned int count;
    double prob;

    feature_t()
    {
        id = 0;
        value = 0;
        type = 0;
        count = 0;
        prob = 0;
    }

    feature_t(int id, int value, short type)
    {
        this->id = id;
        this->value = value;
        this->type = type;
        this->count = 0;
        this->prob = 0;
    }
};

// Structure to hold occurrence of a word, its previous and next words
struct featureOccurrence_t
{
    feature_t feature;
    feature_t *prevfeatures;
    unsigned int sizePrevfeatures;
    feature_t *nextfeatures;
    unsigned int sizeNextfeatures;
    size_t defaultSize;

    featureOccurrence_t(feature_t feature, size_t defaultSize = 65536)
    {
        this->feature = feature;
        this->prevfeatures = NULL;
        this->sizePrevfeatures = 0;
        this->nextfeatures = NULL;
        this->sizeNextfeatures = 0;
        this->defaultSize = defaultSize;
    }

    void addPrevFeature(feature_t prev)
    {
        prev.count = 1;
        if (sizePrevfeatures == 0)
        {
            sizePrevfeatures = 1;
            prevfeatures = (feature_t *)malloc(sizeof(feature_t) * 65536);
            prevfeatures[0] = prev;
        }
        else
        {
            if (prev.id == 0)
            {
                return;
            }
            // if feature already exists, increase count by 1
            for (int i = 0; i < sizePrevfeatures; i++)
            {
                if (prevfeatures[i].id == prev.id)
                {
                    prevfeatures[i].count++;
                    return;
                }
            }
            // if feature does not exist, add it to the list
            sizePrevfeatures++;
            // prevfeatures = (feature_t *)realloc(prevfeatures, sizeof(feature_t) * sizePrevfeatures);
            prevfeatures[sizePrevfeatures - 1] = prev;
        }
    }

    void addNextFeature(feature_t next)
    {
        next.count = 1;
        if (sizeNextfeatures == 0)
        {
            sizeNextfeatures = 1;
            nextfeatures = (feature_t *)malloc(sizeof(feature_t) * this->defaultSize);
            nextfeatures[0] = next;
        }
        else
        {
            // if feature already exists, increase count by 1
            if (next.id == 0)
            {
                return;
            }
            for (int i = 0; i < sizeNextfeatures; i++)
            {
                if (nextfeatures[i].id == next.id)
                {
                    nextfeatures[i].count++;
                    return;
                }
            }
            sizeNextfeatures++;
            // nextfeatures = (feature_t *)realloc(nextfeatures, sizeof(feature_t) * sizeNextfeatures);
            nextfeatures[sizeNextfeatures - 1] = next;
        }
    }

    void addTransition(feature_t prev, feature_t next)
    {
        feature.count++;
        addPrevFeature(prev);
        addNextFeature(next);
    }

    /**
     * @brief Calulate the probability of all features in the featureOccurrence_t
     * @return
     */
    void calculateProbability()
    {
        int totalCount;
        if (sizePrevfeatures == 0)
        {
            return;
        }
        // calculate probability of previous features
        totalCount = 0;
        for (int i = 0; i < sizePrevfeatures; i++)
        {
            totalCount += prevfeatures[i].count;
        }
        for (int i = 0; i < sizePrevfeatures; i++)
        {
            prevfeatures[i].prob = (double)prevfeatures[i].count / (double)totalCount;
        }
        // calculate probability of next features
        totalCount = 0;
        for (int i = 0; i < sizeNextfeatures; i++)
        {
            totalCount += nextfeatures[i].count;
        }
        for (int i = 0; i < sizeNextfeatures; i++)
        {
            nextfeatures[i].prob = (double)nextfeatures[i].count / (double)totalCount;
        }
    }

    /**
     * @brief  print feature occurrence
     * @return
     */
    __device__ __host__ void print()
    {
        printf("Feature: %d %d %d %d\n", feature.id, feature.value, feature.type, feature.count);
        printf("Prev Features: %d\n", sizePrevfeatures);
        for (int i = 0; i < sizePrevfeatures; i++)
        {
            printf("%d %d %d %d\n", prevfeatures[i].id, prevfeatures[i].value, prevfeatures[i].type, prevfeatures[i].count);
        }
        printf("Next Features: %d\n", sizeNextfeatures);
        for (int i = 0; i < sizeNextfeatures; i++)
        {
            printf("%d %d %d %d\n", nextfeatures[i].id, nextfeatures[i].value, nextfeatures[i].type, nextfeatures[i].count);
        }
    }

    /**
     * @brief  assig operator
     * @return
     */
    __device__ __host__ featureOccurrence_t &operator=(const featureOccurrence_t &other)
    {
        this->feature = other.feature;
        this->sizePrevfeatures = other.sizePrevfeatures;
        this->sizeNextfeatures = other.sizeNextfeatures;
        this->defaultSize = other.defaultSize;
        this->prevfeatures = (feature_t *)malloc(sizeof(feature_t) * this->defaultSize);
        this->nextfeatures = (feature_t *)malloc(sizeof(feature_t) * this->defaultSize);
        for (int i = 0; i < sizePrevfeatures; i++)
        {
            this->prevfeatures[i] = other.prevfeatures[i];
        }
        for (int i = 0; i < sizeNextfeatures; i++)
        {
            this->nextfeatures[i] = other.nextfeatures[i];
        }
        return *this;
    }
};

// Structure to hold pointers of feature Occurrences
struct featureOccurrenceCollection_t
{
    featureOccurrence_t *featureOccurences;
    unsigned long size;  // Total number of features
    unsigned long count; // Total number of occurrences

    featureOccurrenceCollection_t()
    {
        this->size = 0;
        this->count = 0;
    }

    int getfeatureIndex(unsigned int id)
    {
        for (int i = 0; i < size; i++)
        {
            // printf("id: %d, value:%d, type:%d\n", featureOccurences[i].feature.id, featureOccurences[i].feature.value, featureOccurences[i].feature.type);
            if (featureOccurences[i].feature.id == id)
            {
                return i;
            }
        }
        // If not found, return -1
        return -1;
    }

    featureOccurrence_t get_feature(unsigned int id)
    {
        int index = getfeatureIndex(id);
        // If not found, return empty feature
        if (index == -1)
        {
            return featureOccurrence_t(feature_t(index, 0, 0));
        }

        return featureOccurences[index];
    }

    void addFeature(featureOccurrence_t featureOccurence)
    {
        if (this->size == 0)
        {
            this->size = 1;
            this->count = 1;
            this->featureOccurences = (featureOccurrence_t *)malloc(sizeof(featureOccurrence_t) * 65536);
            this->featureOccurences[0] = featureOccurence;
        }
        else
        {
            // If feature already exists, increment size
            int index = getfeatureIndex(featureOccurence.feature.id);
            this->count++; // Increment count
            // Feature exists
            if (index != -1)
            {
                printf("Feature already exists\n");
                this->featureOccurences[index] = featureOccurence;
                return;
            }
            else
            {
                // Add new feature to collection
                printf("Adding new feature\n");
                this->size += 1;
                // this->featureOccurences = (featureOccurrence_t *)realloc(this->featureOccurences, newSize);
                this->featureOccurences[size - 1] = featureOccurence;
            }
        }
    }

    /**
     * @brief Calulate the probability of all features in the featureOccurrenceCollection_t
     * @return
     */
    void calculateProbability()
    {
        int totalCount;
        if (size == 0)
        {
            return;
        }
        // calculate probability of features
        totalCount = 0;
        for (int i = 0; i < size; i++)
        {
            totalCount += featureOccurences[i].feature.count;
        }
        for (int i = 0; i < size; i++)
        {
            featureOccurences[i].feature.prob = (double)featureOccurences[i].feature.count / (double)totalCount;
        }
        // calculate probability of features occurencies
        for (int i = 0; i < size; i++)
        {
            featureOccurences[i].calculateProbability();
        }
    }

    /**
     * @brief  print feature occurrence collection
     * @return
     */
    __device__ __host__ void print()
    {
        for (int i = 0; i < size; i++)
        {
            featureOccurences[i].print();
        }
    }

    /**
     * @brief  assig operator
     * @return
     */
    __device__ __host__ featureOccurrenceCollection_t &operator=(const featureOccurrenceCollection_t &other)
    {
        this->size = other.size;
        this->count = other.count;
        this->featureOccurences = (featureOccurrence_t *)malloc(sizeof(featureOccurrence_t) * this->size);
        for (int i = 0; i < size; i++)
        {
            this->featureOccurences[i] = other.featureOccurences[i];
        }
        return *this;
    }
};
