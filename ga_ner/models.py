"""Numba models."""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from itertools import product
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import spacy
from spacy.tokens import Span
from tqdm.auto import tqdm, trange

from ga_ner.linguistic_features import dep_tags, pos_tags
from ga_ner.utils.numba import perfomance_by_doc, slice_doc

if TYPE_CHECKING:
    from spacy.tokens import Doc as SpacyDoc


class Feature:
    """Feature class.
    Represents a linguistic feature of a token.
    """

    def __init__(self, feature: int, value: int):
        self.feature = feature
        self.value = value

    def __iter__(self):
        yield self.feature
        yield self.value

    def to_list(self):
        """Convert to list."""
        return list(self)

    def to_vector(self):
        """Convert to numpy."""
        return np.array(self.to_list(), dtype=np.int64)

    def __str__(self):
        return f"Feature(feature={self.feature}, value={self.value})"

    def __repr__(self):
        return str(self)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Feature):
            return False
        return self.feature == __o.feature and self.value == __o.value

    def __hash__(self) -> int:
        return hash((self.feature, self.value))

    @staticmethod
    def size():
        """Size of the feature."""
        return 2


class Rule:
    """Rule class.
    Represents a rule to identify an entity using token.
    """

    def __init__(
        self, entity: int, tokens: List[Feature], mask: List[bool], score: float = 0.0
    ):
        self.entity = entity
        self.tokens = tokens
        self.mask = mask
        self.score = score

    def __iter__(self):
        """Generator of rule
        Yields: entity, tokens, score
        """
        yield self.entity
        for index, token in enumerate(self.tokens):
            # Feature
            for item in token:
                yield item
            # Include mask
            yield int(self.mask[index])
        yield self.score

    def to_list(self):
        """Convert to list."""
        return list(self)

    @property
    def size(self):
        """Size of the rule."""
        return len(self.tokens)

    def to_vector(self):
        """Convert to numpy."""
        arr = np.zeros(self.size * 3 + 2, dtype=np.float64)
        for i, item in enumerate(self):
            arr[i] = item
        return arr

    @staticmethod
    def from_number_array(array: np.ndarray, score: float = 0.0):
        """Create Rule from numpy array."""
        size: int = array.shape[0]

        entity: int = array[0]
        tokens: List[Feature] = []
        mask: List[bool] = []
        for index in range(1, size, 3):
            feature = Feature(array[index], array[index + 1])
            tokens.append(feature)
            mask.append(bool(array[index + 2]))
        return Rule(entity, tokens, mask, score)

    def __str__(self):
        return f"Rule(entity={self.entity}, tokens={self.tokens}, score={self.score})"

    def __repr__(self) -> str:
        return str(self)


class Entity:
    """Entity class.
    Represents an entity.
    """

    def __init__(self, label: str, text: str, start: int, end: int):
        self.label = label
        self.text = text
        self.start = start
        self.end = end

    @classmethod
    def from_iob(cls, text: str) -> List["Entity"]:
        """Create Entity from IOB format."""
        tokens: List[str] = text.strip().split("\n")
        entity_text = []
        start_span = 0
        end_span = 0
        prev_ent_type = None
        entities: List[Entity] = []
        token_size = len(tokens)
        for i in range(token_size):
            text_, iob_ent = tokens[i].split(" ")
            iob = iob_ent[0]
            ent_type = "O" if iob == "O" else iob_ent[2:]
            if iob == "B":
                if entity_text:
                    end_span = i
                    entities.append(
                        Entity(
                            prev_ent_type, " ".join(entity_text), start_span, end_span
                        )
                    )
                    entity_text = []

                start_span = i
                entity_text.append(text_)
            elif iob == "I":
                entity_text.append(text_)
            elif entity_text and iob == "O":
                end_span = i
                entities.append(
                    Entity(prev_ent_type, " ".join(entity_text), start_span, end_span)
                )
                entity_text = []

            prev_ent_type = ent_type
            if entity_text and i == token_size - 1:
                end_span = i
                entities.append(
                    Entity(ent_type, " ".join(entity_text), start_span, end_span)
                )

        return entities

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Entity):
            return False
        return self.label == __o.label and self.text == __o.text

    def __str__(self):
        return (
            f"Entity(label={self.label}, name={self.text}, "
            f"start={self.start}, end={self.end})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self):
        return self.end - self.start


class Vocabulary:
    """Vocabulary class.
    Represents a vocabulary.
    """

    SPECIAL_STRINGS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    def __init__(self, strings: np.array, include_special: bool = True):
        if include_special:
            str_np = np.concatenate(
                (strings, np.array(self.SPECIAL_STRINGS, dtype=np.str_))
            )
        else:
            str_np = strings

        self.__map_id = {word: index for index, word in enumerate(str_np, 1)}

    @cached_property
    def unk_id(self):
        """Id of unknown word."""
        return self.__map_id.get("[UNK]", 0)

    @cached_property
    def pad_id(self):
        """Id of padding word."""
        return self.__map_id.get("[PAD]", -1)

    @cached_property
    def sep_id(self):
        """Id of separator word."""
        return self.__map_id.get("[SEP]", -2)

    @cached_property
    def cls_id(self):
        """Id of class word."""
        return self.__map_id.get("[CLS]", -3)

    @cached_property
    def mask_id(self):
        """Id of mask word."""
        return self.__map_id.get("[MASK]", -4)

    def __getitem__(self, key: str) -> int:
        return self.__map_id.get(key, self.unk_id)

    def __iter__(self):
        return iter(self.__map_id)

    @classmethod
    def from_list(cls, strings: List[str], include_special: bool = True):
        """Create Vocabulary from list."""
        return cls(np.array(strings), include_special=include_special)

    @classmethod
    def from_iob(cls, path_str: str, encoding: str = "utf-8", threshold=0):
        """Create Vocabulary from IOB file."""
        strings: List[str] = []
        with open(path_str, "r", encoding=encoding) as iob_file:
            lines = iob_file.readlines()

        for line in lines:
            line = line.strip()
            if line:
                strings.append(line.split(" ")[0])

        counter = Counter(strings)
        map_id: Dict[str, int] = {}
        i = 1
        for word, count in counter.items():
            if count >= threshold:
                map_id[word] = i
                i += 1

        return cls.from_list(list(map_id.keys()))

    @classmethod
    def from_file(
        cls, path_str: str, encoding: str = "utf-8", include_special: bool = True
    ):
        """Create Vocabulary from file."""
        strings: List[str] = []
        with open(path_str, "r", encoding=encoding) as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()
            if line:
                strings.append(line)
        return cls.from_list(strings, include_special)

    @classmethod
    def from_spacy(cls, spacy_doc: "SpacyDoc") -> "Vocabulary":
        """Create Vocabulary from spacy doc."""
        return cls(np.array(spacy_doc.vocab.strings))

    def to_file(self, path_str: str, encoding: str = "utf-8"):
        """Save Vocabulary to file."""
        with open(path_str, "w", encoding=encoding) as vocab_file:
            for string in self.__map_id:
                vocab_file.write(f"{string}\n")

    def __len__(self):
        return len(self.__map_id)


class Document:
    """Document class.
    Represents a document to identify entities.
    """

    def __init__(
        self,
        tokens: List[List[Feature]],
        entities: List[Entity],
        vocab: Vocabulary = None,
        vocab_ent: Vocabulary = None,
    ):
        self.tokens = tokens
        self.entities = entities
        self.vocab = vocab
        self.vocab_ent = vocab_ent

    def __iter__(self):
        """
        First Token is meta information about the document.
        Second token is CLS token.
        Last token is SEP token.
        """
        # Add Head Token
        yield len(self.tokens)
        yield len(self.entities)
        yield len(self.vocab)
        yield len(self.vocab_ent)

        # Add CLS token
        yield self.cls_id
        yield self.cls_id
        yield self.cls_id
        yield self.vocab_ent["O"]
        # Add tokens
        entity_index: int = 0
        entity = self.entities[entity_index]
        for index, token in enumerate(self.tokens):
            for feature in token:
                yield self.feature_to_global_index(feature)
            if entity.start <= index < entity.end:
                yield self.vocab_ent[entity.label]
            else:
                yield self.vocab_ent["O"]

            if index == entity.end - 1:
                entity_index += 1
                if entity_index < len(self.entities):
                    entity = self.entities[entity_index]
        # Add SEP token
        yield self.sep_id
        yield self.cls_id
        yield self.cls_id
        yield self.vocab_ent["O"]

    @classmethod
    def from_spacy_document(
        cls, doc: "SpacyDoc", vocab: Vocabulary = None, vocab_ent: Vocabulary = None
    ) -> "Document":
        """Create document from spacy document.
        If vocab is None, it will be created from the spacy model.
        If ents is None, it will be created from the spacy model.
        """
        if vocab is None:
            vocab = Vocabulary(list(doc.vocab.strings))

        tokens = []
        for token in doc:
            tokens.append(
                [
                    Feature(0, pos_tags[token.pos_]),
                    Feature(1, dep_tags[token.dep_]),
                    Feature(2, vocab[token.text]),
                ]
            )

        if vocab_ent is None:
            vocab_ent = Vocabulary(
                np.array(list(set(ent.label_ for ent in doc.ents))),
                include_special=False,
            )

        entities = []
        for ent in doc.ents:
            entities.append(Entity(ent.label_, ent.text, ent.start, ent.end))
        return cls(tokens, entities, vocab, vocab_ent)

    @staticmethod
    @lru_cache(maxsize=16)
    def language(lang: str) -> "spacy.Language":
        """Get spacy language."""
        return spacy.load(lang, exclude=["ner"])

    @classmethod
    def from_iob(
        cls,
        text: str,
        spacy_model: Union[str, spacy.Language] = "es_dep_news_trf",
        vocab: Vocabulary = None,
        vocab_ent: Vocabulary = None,
        add_entities: bool = True,
    ):
        """Create document from sentence in iob format using spacy.
        If vocab is None, it will be created from the spacy model.
        """

        nlp = cls.language(spacy_model) if isinstance(spacy_model, str) else spacy_model
        text_doc = " ".join(map(lambda x: x.split(" ")[0], text.split("\n")))
        doc: "SpacyDoc" = nlp(text_doc)
        if add_entities:
            entities = Entity.from_iob(text)
            # set entities spacy doc
            doc.set_ents([Span(doc, ent.start, ent.end, ent.label) for ent in entities])
        return cls.from_spacy_document(doc, vocab, vocab_ent)

    @classmethod
    def from_text(
        cls,
        text: str,
        spacy_model: str = "es_dep_news_trf",
        vocab: Vocabulary = None,
        vocab_ent: Vocabulary = None,
    ):
        """Create document from sentence in text format using spacy.
        If vocab is None, it will be created from the spacy model.
        """

        nlp = cls.language(spacy_model) if isinstance(spacy_model, str) else spacy_model
        doc: "SpacyDoc" = nlp(text)
        return cls.from_spacy_document(doc, vocab, vocab_ent)

    def rule_from_array(self, array: np.ndarray, entity_type, score: float) -> Rule:
        """Create List of Features from numpy array."""
        feature_list = []
        for index in range(array.shape[0]):
            feature_list.append(self.global_index_to_feature(abs(array[index])))

        return Rule(
            self.vocab_ent[entity_type],
            feature_list,
            array > 0,
            score,
        )

    def to_array(self, entity_label: str = None, exclude_label=False) -> np.ndarray:
        """Convert to numpy array.
        entity_label: if not None, it will convert to others entities to 0.
        Dim = (3 features + 1 label) * (length of tokens +  1 meta token + 2 for [CLS] and [SEP])
        If exclude_label is True, it will not include the label in the array
        then dim = 3 * (length of tokens + 3)
        """
        size: int = len(self.tokens)
        if exclude_label:
            dim = 3 * (size + 3)
        else:
            dim = (3 + 1) * (size + 3)

        tokens: np.ndarray = np.zeros(dim, dtype=np.int32)
        index_token = 0
        for index, item in enumerate(self):
            is_label = index % 4 == 3
            if entity_label is not None and is_label:
                if item == self.vocab_ent[entity_label]:
                    tokens[index_token] = 1
                else:
                    tokens[index_token] = 0
                index_token += 1
            elif not (is_label and exclude_label):
                tokens[index_token] = item
                index_token += 1
        return tokens

    def to_entity_array(self) -> np.ndarray:
        """Convert to numpy array.
        entity_label: if not None, it will convert to others entities to 0.
        Dim = (length of tokens +  1 meta token + 2 for [CLS] and [SEP])
        """
        size: int = len(self.tokens)
        dim = size + 3
        tokens: np.ndarray = np.zeros(dim, dtype=np.int32)
        for entity in self.entities:
            entity_label = entity.label
            tokens[entity.start + 2 : entity.end + 2] = self.vocab_ent[entity_label]

        return tokens

    @lru_cache(maxsize=512)
    def feature_to_global_index(self, feature: Feature) -> int:
        """Get global index of feature."""
        if feature.feature == 0:
            return feature.value

        pos_len = len(pos_tags)
        if feature.feature == 1:
            return feature.value + pos_len

        dep_len = len(dep_tags)
        if feature.feature == 2:
            return feature.value + pos_len + dep_len

    @lru_cache(maxsize=512)
    def global_index_to_feature(self, index: int) -> Feature:
        """Get feature from global index."""
        if index < len(pos_tags):
            return Feature(0, index)

        index -= len(pos_tags)
        if index < len(dep_tags):
            return Feature(1, index)

        index -= len(dep_tags)
        return Feature(2, index)

    @cached_property
    def unk_id(self):
        """Get id of unknown token."""
        return self.vocab.unk_id + len(pos_tags) + len(dep_tags)

    @cached_property
    def mask_id(self):
        """Get mask id."""
        return self.vocab.mask_id + len(pos_tags) + len(dep_tags)

    @cached_property
    def sep_id(self):
        """Get sep id."""
        return self.vocab.sep_id + len(pos_tags) + len(dep_tags)

    @cached_property
    def cls_id(self):
        """Get cls id."""
        return self.vocab.cls_id + len(pos_tags) + len(dep_tags)

    @cached_property
    def pad_id(self):
        """Get pad id."""
        return self.vocab.pad_id + len(pos_tags) + len(dep_tags)

    @cached_property
    def vocab_size(self):
        """Get vocab size."""
        return len(self.vocab) + len(pos_tags) + len(dep_tags) + 1

    def sampling(self, windows=5, add_special=True) -> List[Tuple[int, int, int]]:
        """Generate representation sampling using combinations of features
        Return list of tuples wich length is windows
        """
        tokens: List[List[Feature]] = None
        if add_special:
            tokens = (
                [[Feature(2, self.vocab.cls_id)]]
                + self.tokens
                + [[Feature(2, self.vocab.sep_id)]]
            )

        else:
            tokens = self.tokens

        if windows > len(tokens):
            windows = len(tokens)

        for index in range(len(tokens) - windows + 1):
            values_by_feature: List[List[int]] = []
            for i in range(index, windows + index):
                values_by_feature.append(
                    [self.feature_to_global_index(feature) for feature in tokens[i]]
                )
            windows_combination = product(*values_by_feature)
            for combination in windows_combination:
                yield combination


@dataclass
class Corpus:
    """Corpus class.
    Represents a corpus to identify entities.
    """

    documents: List[Document]
    entities: Dict[str, List[int]] = field(init=False)
    vocab_ent: Vocabulary = None

    def __post_init__(self):
        """Post init."""
        # Entities by document
        self.entities = defaultdict(list)
        for index, doc in enumerate(self.documents):
            for entity in doc.entities:
                self.entities[entity.label].append(index)

        # clean indices, remove duplicates
        for label, indices in self.entities.items():
            self.entities[label] = list(set(indices))

        # Add vocab entities to documents)
        if self.vocab_ent is None:
            all_entities = list(self.entities.keys())
            self.vocab_ent = Vocabulary.from_list(all_entities, include_special=False)
            for doc in self.documents:
                doc.vocab_ent = self.vocab_ent

    def from_text(self, text: str):
        """Create corpus from text."""
        pass

    def from_array(self, array: np.ndarray):
        """Create corpus from numpy array."""
        pass

    @classmethod
    def from_iob_file(cls, file_path: str, encoding: str = "utf-8", **kwargs):
        """Create corpus from iob file.
        kwargs: additional arguments for Document.from_iob
        """
        with open(file_path, "r", encoding=encoding) as file:
            text = file.read()
        for sentence in text.split("\n\n"):
            yield Document.from_iob(sentence, **kwargs)

    @classmethod
    def from_text_file(cls, file_path: str, encoding: str = "utf-8", **kwargs):
        """Create corpus from text file.
        kwargs: additional arguments for Document.from_text
        """
        with open(file_path, "r", encoding=encoding) as file:
            text = file.read().strip()
        for sentence in text.split("\n"):
            yield Document.from_text(sentence, **kwargs)

    @classmethod
    def from_file(
        cls, file_path: str, type_files: str = "iob", encoding: str = "utf-8", **kwargs
    ):
        """Create corpus from file.
        kwargs: additional arguments for from_iob_file or from_text_file
        """
        docs: Iterable[Document] = None
        if type_files == "iob":
            docs = cls.from_iob_file(file_path, encoding, **kwargs)
        elif type_files == "text":
            docs = cls.from_text_file(file_path, encoding, **kwargs)
        else:
            raise ValueError("Type file not supported")
        return cls(list(docs))

    @classmethod
    def from_spacy_docs(
        cls, spacy_docs: Iterable["SpacyDoc"], total_samples=None, **kwargs
    ):
        """Create corpus from spacy docs."""
        return cls(
            [
                Document.from_spacy_document(doc, **kwargs)
                for doc in tqdm(
                    spacy_docs,
                    total=total_samples,
                )
            ],
            vocab_ent=kwargs.get("vocab_ent"),
        )

    def to_text_array(
        self,
        entity_label: str = None,
        exclude_label=False,
        max_size_doc: int = 172,
        input_filename="input.txt",
        target_filename="target.txt",
        metadata_filename="metadata.txt",
        encoding="utf-8",
    ):
        """Create files with text from documents.
        Create a file to inputs, targets and other file to metadata.
        """
        with open(input_filename, "w", encoding=encoding) as input_file, open(
            target_filename, "w", encoding=encoding
        ) as target_file, open(
            metadata_filename, "w", encoding=encoding
        ) as metadata_file:
            for _, document in enumerate(self.documents):

                if len(document.entities) == 0:
                    continue

                document_array = document.to_array(entity_label, exclude_label).reshape(
                    -1, 4
                )
                max_size = min(max_size_doc, document_array.shape[0])

                input_data = np.zeros((max_size_doc, 3), dtype=np.float32)
                target = np.zeros((max_size_doc, 1), dtype=np.int32)
                meta = np.zeros((4), dtype=np.float32)

                meta[:] = document_array[0]
                input_data[: max_size - 1] = document_array[1:max_size, :3]
                target[: max_size - 1] = document_array[1:max_size, 3:]

                for line in input_data:
                    input_file.write(",".join([str(x) for x in line]))
                    input_file.write(" ")

                for line in target:
                    target_file.write(str(line[0]))
                    target_file.write(" ")

                metadata_file.write(",".join([str(x) for x in meta]))

                input_file.write("\n")
                target_file.write("\n")
                metadata_file.write("\n")


@dataclass
class GANER:
    """GANER class.
    Represents a Genetic Algothitim NER model.
    If chromosome's values is positive, then it includes the Entity.
    """

    data: np.ndarray
    # target: np.ndarray
    # meta_data: np.ndarray
    map_inv_entity: Dict[int, str]
    n_population: int
    max_len: int
    mask_id: int
    unknown_id: int
    ml_model: pl.Trainer
    n_top: int
    population: np.ndarray = None
    population_fitness: np.ndarray = None
    random_state: int = None
    threshold: float = 0.5
    select: Callable = None
    fitness: Callable = None
    num_threads: int = None

    def create_gen(self, cromosome: np.ndarray):
        """Create gen.
        Run ml model with cromosome and return gen.
        Select gen with ramdom with probability distribution.
        is_entity: True if gen is entity. If it's False the value gen is negative.
        """
        import torch
        from torch.utils.data import DataLoader

        batch_size = 32
        prob_list = self.ml_model.predict(
            self.ml_model.model,
            DataLoader(
                torch.from_numpy(cromosome.astype(np.int)), batch_size=batch_size
            ),
        )
        gen_arr = np.empty(cromosome.shape[0], dtype=np.int)

        index = 0
        for prob_bacth in prob_list:
            for prob in prob_bacth:
                top10 = prob.topk(self.n_top).indices
                gen_arr[index] = np.random.choice(top10.numpy(), 1)
                index += 1

        return gen_arr

    def init_population(
        self,
        population: np.ndarray,
        population_fitness: np.ndarray,
        base_population: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize population.
        First token is size of rule.
        Second token is its group.
        Third token is to indicate entity type.
        Fourth token is mask.
        """
        # Copy base population
        if base_population is not None:
            pop_num_indiv = min(population.shape[0], base_population.shape[0])
            pop_num_features = min(base_population.shape[1], population.shape[1])
            population[:pop_num_indiv, :pop_num_features] = base_population[
                :pop_num_indiv, :pop_num_features
            ]
        # Add mask to population without gene
        filter_add_mask = np.where(population[:, 3] == 0, True, False)
        if filter_add_mask.sum() > 0:
            entity_types = np.random.choice(
                range(1, 4), size=population.shape[0], replace=True
            )
            population[filter_add_mask, 3] = self.mask_id
            num_gen = 1
            population[filter_add_mask, 0] = num_gen
            population[filter_add_mask, 1] = -100
            population[filter_add_mask, 2] = entity_types[filter_add_mask]

        # Create genes to first feature is mask
        filter_create_gene = np.where(
            population[:, 3] == self.mask_id, True, False
        ).astype(int)
        population[filter_create_gene, 3] = self.create_gen(
            population[filter_create_gene, 3:4]
        )
        # Calculate fitness
        population_fitness[:] = self._fitness(population)[:]
        return population_fitness

    def mutate(
        self, individual: np.ndarray, individual1: np.ndarray, individual2: np.ndarray
    ) -> np.ndarray:

        """Mutate individual.
        Select random if add, remove or change a gen.
        If size of individual is more than max_len, remove last feature.
        If size between 3 and max_len, Select random feature and change it to masked.
        The masked feature is replaced with feature from create_gen.
        To optimize the process, create_gen is called with False and True.
        """
        try:
            # TODO Model to select operation given it improves performance.
            size_individual: int = int(individual[0])
            # 0 = add, 1 = remove, 2 = change 3 = change first, 4 = change last 5 = remove first, 6 = remove last
            operation = np.random.randint(0, 3)
            index_feature = 0
            if size_individual == 1 and operation == 1:
                operation = 0
                index_feature = size_individual + 3
            if size_individual == 1 and operation == 2:
                index_feature = 3
            elif operation == 0 and size_individual + 3 >= self.max_len:
                operation = 1
                index_feature = self.max_len - 1

            if operation == 0:
                size_individual += 1

            if not index_feature:
                index_feature = np.random.randint(3, 3 + size_individual)

            try:
                if operation == 0:
                    individual1[0] = size_individual
                    individual2[0] = size_individual

                    for index in range(3 + size_individual - 1, index_feature, -1):
                        next_val = individual[index - 1]
                        individual1[index] = next_val
                        individual2[index] = next_val

                elif operation == 1:
                    size_individual -= 1
                    individual1[0] = size_individual
                    individual2[0] = size_individual
                    for index in range(index_feature, 3 + size_individual):
                        next_val = individual[index + 1]
                        individual1[index] = next_val
                        individual2[index] = next_val

                    individual1[3 + size_individual] = 0
                    individual2[3 + size_individual] = 0
            except IndexError:
                raise

            if operation in [0, 2]:
                new_size: int = int(individual1[0]) + 3
                individual1[index_feature] = self.mask_id
                gen = self.create_gen(np.abs(individual1[3:new_size]).reshape(1, -1))
                individual1[index_feature] = gen
                individual2[index_feature] = -gen
        except IndexError as e:
            raise
        return individual1, individual2

    def crossover(
        self,
        individual1: np.ndarray,
        individual2: np.ndarray,
        individual1_new: np.ndarray,
        individual2_new: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover individuals.
        Point of crossover is random but it is the min len between individuals.
        """
        max_len = min(individual1[0], individual2[0])
        point_crossover = 3 + 1
        individual1_new = np.zeros(individual1.shape, dtype=np.float32)
        individual2_new = np.zeros(individual2.shape, dtype=np.float32)
        if max_len > 1:
            point_crossover = np.random.randint(1, max_len) + 3

        individual1_new[3:] = np.concatenate(
            (individual1[3:point_crossover], individual2[point_crossover:])
        )
        individual2_new[3:] = np.concatenate(
            (individual2[3:point_crossover], individual1[point_crossover:])
        )
        individual1_new[0] = (individual1_new[3:] != 0).sum()
        individual2_new[0] = (individual2_new[3:] != 0).sum()
        return individual1_new, individual2_new

    def _fitness(self, population: np.ndarray) -> np.ndarray:
        """Calculate fitness of individual."""
        return self.fitness(
            population,
            self.data["ALL"]["input"],
            self.data["ALL"]["target"][0],
            self.data["ALL"]["meta"],
            self.unknown_id,
            self.num_threads,
        ).base

    def train_step(
        self,
        parent_population: np.ndarray,
        population_fitness: np.ndarray,
        offspring_population: np.ndarray,
        offspring_fitness: np.ndarray,
        n_population: int,
    ):
        """Train step model.
        max_iter: maximum number of iterations.
        """
        index_population = 0

        while index_population < n_population:
            individual_selected = parent_population[index_population]
            individual1 = individual_selected.copy()
            individual2 = individual_selected.copy()
            self.mutate(individual_selected, individual1, individual2)

            offspring_population[index_population * 2] = individual1
            offspring_population[index_population * 2 + 1] = individual2

            index_population += 1

        # Calculate fitness of offspring
        offspring_fitness[:] = self._fitness(offspring_population)[:]
        population_fitness[:] = self._fitness(parent_population)[:]

        self.select(
            parent_population,
            population_fitness,
            offspring_population,
            offspring_fitness,
            n_population,
            threshold=self.threshold,
            num_threads=self.num_threads,
        )

        return parent_population

    def train(
        self,
        max_iter: int,
        tol=10,
        *,
        base_population=None,
        num_islands=1,
        num_threads=1,
    ):
        """Train step model.
        max_iter: maximum number of iterations.
        """
        self.num_threads = num_threads
        np.random.seed(self.random_state)

        parent_population = np.zeros(
            (self.n_population, self.max_len), dtype=np.float32
        )
        population_fitness = np.zeros((self.n_population,), dtype=np.float32)
        self.init_population(parent_population, population_fitness, base_population)
        # Variables to control the convergence
        self.population = np.zeros((self.n_population, self.max_len), dtype=np.float32)
        n_not_improve = 0
        mean_best_fitness = 0
        i = 0
        pbar = trange(max_iter)
        offspring_population = np.zeros(
            (self.n_population * 2, self.max_len), dtype=np.float32
        )
        offspring_fitness = np.zeros((self.n_population * 2,), dtype=np.float32)

        for i in pbar:

            step_range = self.n_population // num_islands
            for index_island in range(0, self.n_population, step_range):
                self.train_step(
                    parent_population[index_island : index_island + step_range],
                    population_fitness[index_island : index_island + step_range],
                    offspring_population[
                        index_island * 2 : (index_island + step_range) * 2
                    ],
                    offspring_fitness[
                        index_island * 2 : (index_island + step_range) * 2
                    ],
                    n_population=min(index_island + step_range, self.n_population)
                    - index_island,
                )

            # Migration
            if num_islands > 1 and i % 10 == 0:
                for index_island in range(num_islands):
                    index_migration = (
                        np.random.randint(0, step_range) + index_island * step_range
                    )
                    index_accept = (
                        np.random.randint(0, step_range)
                        + (index_island + 1) * step_range
                    )
                    if index_accept >= self.n_population:
                        index_accept = index_accept - self.n_population
                    if (
                        population_fitness[index_migration]
                        > population_fitness[index_accept]
                    ):
                        parent_population[index_accept, :] = parent_population[
                            index_migration
                        ].copy()
                        population_fitness[index_accept] = population_fitness[
                            index_migration
                        ].copy()

            mean_fitness = population_fitness.mean()
            max_curr_fitness = population_fitness.max()
            if mean_fitness > 1:
                print(mean_fitness)
            pbar.set_description(
                f"It:{i:02d}, Fitness:{mean_fitness:.4f}, Best Fitness:{max_curr_fitness:.4f}"
            )
            if (mean_fitness) > (mean_best_fitness):
                best_fitness = mean_fitness
                mean_best_fitness = mean_fitness
                self.population = parent_population.copy()
                n_not_improve = 0
                self.population[:, 1] = -100
                self.save()
            else:
                n_not_improve += 1
                if n_not_improve > tol:
                    break

    def save(self):
        """Remove duplicates from population and save to file."""
        self.population = np.unique(self.population, axis=0)
        # remove if all are less than 0
        self.population = self.population[self.population[:, 0] > 0]
        self.population_fitness = self._fitness(self.population)
        np.save(
            f"best_population.npy",
            self.population[self.population_fitness > 0],
        )
        np.save(
            f"best_fitness.npy",
            self.population_fitness[self.population_fitness > 0],
        )

    def predict(self, data: np.ndarray):
        """Predict."""
        pass
