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

    def __init__(self, feature: int, value: Union[List[float], float]):
        self.feature = int(feature)
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

    def to_dict(self, vocab: "Vocabulary", vocab_entities: "Vocabulary"):
        """Convert to text.
        label is the entity.
        pattern is the rule.
        """
        features_dict = {
            0: "POS",
            1: "DEP",
            2: "TEXT",
        }

        inverted_dict = {
            0: {v: k for k, v in pos_tags.items()},
            1: {v: k for k, v in dep_tags.items()},
            2: vocab.inverse(),
        }
        vocab_inv = vocab_entities.inverse()
        return {
            "label": vocab_inv[self.entity],
            "pattern": [
                {
                    features_dict[token.feature]: inverted_dict[token.feature][
                        token.value
                    ],
                    "is_entity": mask,
                }
                for token, mask in zip(self.tokens, self.mask)
            ],
        }

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
        return f"Rule(entity={self.entity}, tokens={self.tokens}, score={self.score}, mask={self.mask})"

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

    def items(self):
        """Return items."""
        return self.__map_id.items()

    def inverse(self) -> Dict[int, str]:
        """Id to string."""
        return {v: k for k, v in self.__map_id.items()}

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


@dataclass
class Document:
    """Document class.
    Represents a document to identify entities.
    """

    tokens: List[List[Feature]]
    entities: List[Entity]
    vocab: Vocabulary = None
    vocab_ent: Vocabulary = None
    embeding_func: Callable = None
    emb_size: int = 1

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
                if self.embeding_func:
                    yield feature.value
                else:
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
        cls,
        doc: "SpacyDoc",
        vocab: Vocabulary = None,
        vocab_ent: Vocabulary = None,
        *,
        embeding_func: Callable = None,
        emb_size: int = 1,
    ) -> "Document":
        """Create document from spacy document.
        If vocab is None, it will be created from the spacy model.
        If ents is None, it will be created from the spacy model.
        """
        if vocab is None:
            vocab = Vocabulary(list(doc.vocab.strings))

        tokens = []
        for token in doc:
            if embeding_func is None:
                tokens.append(
                    [
                        Feature(0, vocab[token.pos_]),
                        Feature(1, vocab[token.dep_]),
                        Feature(2, vocab[token.text]),
                    ]
                )
            else:
                tokens.append(
                    [
                        Feature(0, embeding_func(token.pos_)),
                        Feature(1, embeding_func(token.dep_)),
                        Feature(2, embeding_func(token.text)),
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
        return cls(
            tokens,
            entities,
            vocab,
            vocab_ent,
            embeding_func=embeding_func,
            emb_size=emb_size,
        )

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
            entity_type,
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

        size + 2 ([CLS] and [SEP]) are embedding
        """
        size: int = len(self.tokens)
        if exclude_label:
            dim = 3 * ((size + 2) * self.emb_size) + 3
        else:
            dim = 3 * ((size + 2) * self.emb_size) + 3 + (size + 3)

        tokens: np.ndarray = np.zeros(dim, dtype=np.float32)
        index_token = 0
        # Meta data has not embedding
        limit_meta = 3 if exclude_label else 4
        num_entities = 0
        for index, item in enumerate(self):
            is_label = index % 4 == 3
            if entity_label is not None and is_label:
                if item == self.vocab_ent[entity_label]:
                    tokens[index_token * self.emb_size] = 1
                else:
                    tokens[index_token * self.emb_size] = 0
                index_token += 1
            elif not (is_label and exclude_label):
                start_slice = (
                    (index_token - limit_meta - num_entities) * self.emb_size
                    + limit_meta
                    + num_entities
                )
                if hasattr(item, "__iter__"):
                    # Minus number of entities
                    end_slice = start_slice + self.emb_size
                    tokens[start_slice:end_slice] = item[:]
                elif index >= limit_meta:
                    # Number of entities
                    tokens[start_slice] = item
                    num_entities += 1
                else:
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

    @property
    def text(self) -> str:
        """Get text."""
        for token in self.tokens:
            print(token[2], self.vocab.inverse()[token[2].value])
        return " ".join([self.vocab.inverse()[token[2].value] for token in self.tokens])

    @cached_property
    def unk_id(self):
        """Get id of unknown token."""
        if self.embeding_func is None:
            return self.vocab.unk_id + len(pos_tags) + len(dep_tags)
        return self.embeding_func("[UNK]")

    @cached_property
    def mask_id(self):
        """Get mask id."""
        if self.embeding_func is None:
            return self.vocab.mask_id + len(pos_tags) + len(dep_tags)
        return self.embeding_func("[MASK]")

    @cached_property
    def sep_id(self):
        """Get sep id."""
        if self.embeding_func is None:
            return self.vocab.sep_id + len(pos_tags) + len(dep_tags)
        return self.embeding_func("[SEP]")

    @cached_property
    def cls_id(self):
        """Get cls id."""
        if self.embeding_func is None:
            return self.vocab.cls_id + len(pos_tags) + len(dep_tags)
        return self.embeding_func("[CLS]")

    @cached_property
    def pad_id(self):
        """Get pad id."""
        if self.embeding_func is None:
            return self.vocab.pad_id + len(pos_tags) + len(dep_tags)
        return self.embeding_func("[PAD]")

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
    embeding_func: Callable = None
    emb_size: int = 1

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
            [Document.from_spacy_document(doc, **kwargs) for doc in spacy_docs],
            vocab_ent=kwargs.get("vocab_ent"),
            emb_size=kwargs.get("emb_size"),
        )

    def to_text_array(
        self,
        entity_label: str = None,
        exclude_label=False,
        max_size_doc: int = 172,
        encoding="utf-8",
        *,
        input_filename="input.txt",
        target_filename="target.txt",
        metadata_filename="metadata.txt",
        confidence_docs=100,
    ):
        """Create files with text from documents.
        Create a file to inputs, targets and other file to metadata.
        """
        with open(input_filename, "w", encoding=encoding) as input_file, open(
            target_filename, "w", encoding=encoding
        ) as target_file, open(
            metadata_filename, "w", encoding=encoding
        ) as metadata_file:
            for index_doc, document in enumerate(self.documents):
                if len(document.entities) == 0:
                    continue

                document_array = document.to_array(entity_label, exclude_label)
                max_size = int(min(max_size_doc, document_array[0] + 2))
                emb_size = self.emb_size * 3

                input_data = np.zeros(max_size * emb_size, dtype=np.float32)
                target = np.zeros(max_size, dtype=np.int32)
                meta = np.zeros(4, dtype=np.int32)
                meta[:4] = document_array[:4]
                # Confidence in labeled document
                meta[2] = confidence_docs
                meta[3] = index_doc
                document_array = document_array[4:].reshape(-1, emb_size + 1)
                input_data[:] = document_array[:max_size, :emb_size].reshape(-1)
                target[:] = document_array[:max_size, emb_size:].reshape(-1)

                input_file.write(",".join([str(x) for x in input_data]))
                target_file.write(",".join([str(x) for x in target]))
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

    map_inv_entity: Dict[int, str]
    n_population: int
    max_len: int
    mask_id: int
    unknown_id: int
    n_top: int
    candidate_words: np.ndarray
    population: np.ndarray = None
    population_fitness: np.ndarray = None
    random_state: int = None
    threshold: float = 0.5
    select: Callable = None
    fitness: Callable = None
    num_threads: int = None
    num_features: int = 1
    embedding_size: int = 1

    def __post_init__(self):
        """Initialize random seed."""
        np.random.seed(self.random_state)

    def create_gen(
        self,
        individual: np.ndarray,
        position: int,
        candidate_words: np.ndarray,
        size_candidate: int,
        is_entity: float = 1.0,
        embedding_size: int = 1,
    ):
        """Create gen.
        Select random word from candidate words.

        Args:
        -----
            individual (np.ndarray): Individual.
            position: (int): Position of gen.
            candidate_words (np.ndarray): Candidate words.
            size_candidate (int): Size of candidate words.
            is_entity (float): Value to indicate if is entity.

        Returns:
        --------
            individual (np.ndarray): Indivivual with new gen.
        """
        start_position = position * (embedding_size + 2) + 3
        random_index = np.random.randint(0, size_candidate)
        individual[start_position] = is_entity
        individual[
            start_position + 1 : start_position + embedding_size + 2
        ] = candidate_words[random_index]
        return individual

    def create_genes(
        self,
        individuals: np.ndarray,
        positions: np.ndarray,
        num_genes: int,
        candidate_words: np.ndarray,
        is_entities: np.ndarray,
        embedding_size=1,
    ):
        """Create genes.

        Args:
        -----
            individuals (np.ndarray): Individual.
            positions: (np.ndarray): Position of gen.
            num_genes (int): Number of genes to be created.
            candidate_words (np.ndarray): Candidate words.

        Returns:
        --------
            individuals (np.ndarray): Individuals with new genes.
        """
        size_candidate = candidate_words.shape[0]
        for i in range(num_genes):
            self.create_gen(
                individuals[i],
                positions[i],
                candidate_words,
                size_candidate,
                is_entity=is_entities[i],
                embedding_size=embedding_size,
            )
        return individuals

    def init_population(
        self,
        input_data,
        target,
        meta,
        population: np.ndarray,
        population_fitness: np.ndarray,
        base_population: np.ndarray = None,
        embedding_size: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize population.
        First token is size of rule.
        Second token is its fitness value.
        Third token is to indicate entity type.
        Next token are segments of genes.
        """
        # Copy base population
        if base_population is not None:
            pop_num_indiv = min(population.shape[0], base_population.shape[0])
            pop_num_features = min(base_population.shape[1], population.shape[1])
            population[:pop_num_indiv, :pop_num_features] = base_population[
                :pop_num_indiv, :pop_num_features
            ]
        # It added a mask to population without gene
        # 3 indicate entity type, if 0 then the first chromosome is empty
        filter_add_mask: np.ndarray = np.where(population[:, 3] == 0, True, False)
        num_empty_genes = filter_add_mask.sum()
        if num_empty_genes > 0:
            # Select random entity type
            entity_types = np.random.choice(
                range(1, 4), size=population.shape[0], replace=True
            )
            population[filter_add_mask, 0] = 1
            population[filter_add_mask, 1] = 0
            population[filter_add_mask, 2] = entity_types[filter_add_mask]
            population[filter_add_mask] = self.create_genes(
                population[filter_add_mask],
                np.zeros(num_empty_genes, dtype=np.int32),
                num_empty_genes,
                self.candidate_words,
                is_entities=entity_types[filter_add_mask],
                embedding_size=embedding_size,
            )

        # Calculate fitness
        population_fitness[:] = self._fitness(
            input_data,
            target,
            meta,
            population,
            num_features=3,
            embedding_size=embedding_size,
        )[:]
        return population_fitness

    def mutate(
        self,
        individual: np.ndarray,
        individual1: np.ndarray,
        individual2: np.ndarray,
        entity_island: int = None,
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
            # 0 = add, 1 = remove, 2 = change 3 = change first
            embedding_size: int = self.embedding_size + 2
            max_features = (self.max_len - 3) // embedding_size
            operation = np.random.randint(0, 3)
            index_feature = 0

            if individual1[3 + (size_individual - 1) * embedding_size + 2] == 0:
                pass
            # Validate correct operation
            if size_individual == 1 and operation == 1:
                operation = 0
                index_feature = size_individual
            if size_individual == 1 and operation == 2:
                index_feature = 0
            elif operation == 0 and size_individual >= max_features:
                operation = 1
                index_feature = max_features - 1
            # Update size of individual for add
            if operation == 0:
                size_individual += 1

            # Select random feature
            if not index_feature:
                index_feature = np.random.randint(size_individual)

            if (
                individual1[0] == 2
                and individual1[3 + (int(individual[0]) - 1) * embedding_size + 2] == 0
            ):
                pass
            try:
                if operation == 0:
                    individual1[0] = size_individual
                    individual2[0] = size_individual
                    # TODO Test to add a new feature
                    for index in range(size_individual - 1, index_feature, -1):
                        start_index = index * embedding_size + 3
                        end_index = start_index + embedding_size
                        next_start_index = (index - 1) * embedding_size + 3
                        next_end_index = next_start_index + embedding_size
                        next_val = individual[next_start_index:next_end_index]
                        individual1[start_index:end_index] = next_val.copy()
                        individual2[start_index:end_index] = next_val.copy()

                elif operation == 1:
                    size_individual -= 1
                    individual1[0] = size_individual
                    individual2[0] = size_individual
                    for index in range(
                        index_feature,
                        size_individual,
                    ):
                        start_index = index * embedding_size + 3
                        end_index = start_index + embedding_size
                        next_start_index = (index + 1) * embedding_size + 3
                        next_end_index = next_start_index + embedding_size
                        next_val = individual[next_start_index:next_end_index]
                        individual1[start_index:end_index] = next_val.copy()
                        individual2[start_index:end_index] = next_val.copy()

                    individual1[3 + size_individual * embedding_size :] = 0
                    individual2[3 + size_individual * embedding_size :] = 0
            except IndexError:
                raise

            if operation in [0, 2]:
                # new_size: int = int(individual1[0])
                # individual1[0] = new_size
                # individual2[0] = new_size
                # TODO Create function to change a segment
                self.create_gen(
                    individual1,
                    index_feature,
                    self.candidate_words,
                    self.candidate_words.shape[0],
                    is_entity=entity_island or individual1[2],
                    embedding_size=self.embedding_size,
                )
                # Update individual2 with new gene and change the first gene's segment to 0 (not entity)
                start_index = index_feature * embedding_size + 3
                end_index = start_index + embedding_size
                individual2[start_index:end_index] = individual1[
                    start_index:end_index
                ].copy()
                individual2[start_index] = 0
        except IndexError as e:
            raise

        if individual1[3 + (int(individual1[0]) - 1) * embedding_size + 2] == 0:
            pass

        if individual2[3 + (int(individual2[0]) - 1) * embedding_size + 2] == 0:
            pass
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

    def _fitness(
        self,
        input_data,
        target,
        meta,
        population: np.ndarray,
        num_features: int,
        embedding_size: int,
    ) -> np.ndarray:
        """Calculate fitness of individual."""
        return self.fitness(
            input_data, target, meta, population, num_features, embedding_size
        )

    def train_step(
        self,
        parent_population: np.ndarray,
        population_fitness: np.ndarray,
        offspring_population: np.ndarray,
        offspring_fitness: np.ndarray,
        n_population: int,
        *,
        docs: np.ndarray,
        target: np.ndarray,
        meta: np.ndarray,
        num_features: int = 3,
        embedding_size: int = 1,
        island_number: int | None = None,
        entity_island: int | None = None,
    ):
        """Train step model.
        max_iter: maximum number of iterations.
        """
        index_population = 0

        while index_population < n_population:
            individual_selected = parent_population[index_population]
            individual1 = individual_selected.copy()
            individual2 = individual_selected.copy()
            if individual1[0] >= 2 and individual1[39] == 0:
                raise ValueError("Individual 1 is not valid")
            self.mutate(
                individual_selected,
                individual1,
                individual2,
                entity_island=entity_island,
            )

            offspring_population[index_population * 2] = individual1
            offspring_population[index_population * 2 + 1] = individual2

            index_population += 1

        # Calculate fitness of offspring
        offspring_fitness[:] = self._fitness(
            docs,
            target=target,
            meta=meta,
            population=offspring_population,
            num_features=num_features,
            embedding_size=embedding_size,
        )[:]
        population_fitness[:] = self._fitness(
            docs,
            target=target,
            meta=meta,
            population=parent_population,
            num_features=num_features,
            embedding_size=embedding_size,
        )[:]

        for p in parent_population:
            if p[0] >= 2 and p[39] == 0:
                raise ValueError("Individual is not valid")

        for p in offspring_population:
            if p[0] >= 2 and p[39] == 0:
                raise ValueError("Individual is not valid")

        self.select(
            parent_population,
            population_fitness,
            offspring_population,
            offspring_fitness,
            n_population,
            threshold=self.threshold,
            num_threads=self.num_threads,
        )

        for p in parent_population:
            if p[0] >= 2 and p[39] == 0:
                raise ValueError("Individual is not valid")

        return parent_population

    def train(
        self,
        input_data: np.ndarray,
        target: np.ndarray,
        meta: np.ndarray,
        max_iter: int,
        tol=10,
        *,
        base_population=None,
        num_islands=1,
        num_threads=1,
        save_path=None,
        sufix="",
    ):
        """Train step model.
        max_iter: maximum number of iterations.
        """
        self.num_threads = num_threads

        parent_population = np.zeros(
            (self.n_population, self.max_len), dtype=np.float32
        )
        population_fitness = np.zeros((self.n_population,), dtype=np.float32)
        self.init_population(
            input_data,
            target,
            meta,
            parent_population,
            population_fitness,
            base_population,
            self.embedding_size,
        )
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
        # TODO Number of islands have to be multiple of number entities
        index_docs_arr = np.zeros(
            (num_islands, int(target.shape[0] * 0.6)), dtype=np.int32
        )
        for i in range(num_islands):
            index_docs_arr[i] = np.random.choice(
                target.shape[0], size=int(target.shape[0] * 0.6), replace=False
            )
        for i in pbar:
            step_range = self.n_population // num_islands
            # TODO Parralelize for each island
            # TODO ADD entity by island
            entity_island_id = 0
            for index_island in range(0, self.n_population, step_range):
                # Each island has a different target to documents
                entity_island_id = (entity_island_id % num_islands) + 1
                # index_docs = index_docs_arr[index_island // step_range]
                index_docs = np.where((target == entity_island_id).any(axis=1))[0]
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
                    docs=input_data[index_docs],
                    target=target[index_docs],
                    meta=meta[index_docs],
                    num_features=self.num_features,
                    embedding_size=self.embedding_size,
                    entity_island=entity_island_id,
                )

            # Migration
            if num_islands > 1 and i % 10 == 0 and i > 0:
                # Select random individuals from each island to migrate
                migration_indexes = np.random.randint(0, step_range, size=num_islands)
                migration_individuals = np.zeros(
                    (num_islands, self.max_len), dtype=np.float32
                )
                migration_fitness = np.zeros((num_islands,), dtype=np.float32)

                for index_island in range(num_islands):
                    migration_index = (
                        migration_indexes[index_island] + index_island * step_range
                    ).item()
                    migration_individuals[index_island] = parent_population[
                        migration_index
                    ].copy()
                    migration_fitness[index_island] = population_fitness[
                        migration_index
                    ].copy()

                # Select random individuals from each island to accept migration
                random_islands = np.random.choice(
                    num_islands, size=num_islands, replace=False
                )
                for index_island in range(num_islands):
                    next_island = random_islands[(index_island + 1) % num_islands]
                    accepted_index = (
                        migration_indexes[next_island] + next_island * step_range
                    ).item()

                    parent_population[accepted_index, :] = migration_individuals[
                        random_islands[index_island]
                    ].copy()
                    population_fitness[accepted_index] = migration_fitness[
                        random_islands[index_island]
                    ].copy()

            mean_fitness = population_fitness.mean()
            max_curr_fitness = population_fitness.max()
            if mean_fitness > 1:
                print(mean_fitness)
            pbar.set_description(
                f"It:{i:02d}, Fitness:{mean_fitness:.4f}, Best Fitness:{max_curr_fitness:.4f}"
            )
            if (mean_fitness) > (mean_best_fitness):
                mean_best_fitness = mean_fitness
                self.population = parent_population.copy()
                n_not_improve = 0
                self.population[:, 1] = 0
                self.save(input_data, target, meta, save_path, sufix)
            else:
                n_not_improve += 1
                if n_not_improve > tol:
                    break

    def save(
        self, docs, target: np.ndarray, meta: np.ndarray, save_path=None, sufix=""
    ):
        """Save model.
        Remove duplicates from population and save to file.
        """
        self.population = np.unique(self.population, axis=0)
        # remove if all are less than 0
        self.population = self.population[self.population[:, 0] > 0]
        self.population_fitness = self._fitness(
            docs, target, meta, self.population, self.num_features, self.embedding_size
        )

        theshold = 0.5
        fitness_filter = self.population_fitness > theshold
        # print(f"Saving {fitness_filter.sum()} individuals")

        if save_path is None:
            save_path = "."

        np.save(
            f"{save_path}/best_population{sufix}.npy", self.population[fitness_filter]
        )
        np.save(
            f"{save_path}/best_fitness{sufix}.npy",
            self.population_fitness[fitness_filter],
        )

    def predict(self, data: np.ndarray):
        """Predict."""
        pass
