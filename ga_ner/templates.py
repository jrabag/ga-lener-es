"""Models to create the templates for the GA NER.
Templates have entities that can be used to replace with entities that are same type.
The entities can be collected from corpus, sinonimous.
The sinonimous can be collected from vectorized models, ontologies, or dictionaries.
"""
from itertools import product
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import spacy

from ga_ner.models import Entity

BASE_PATH = Path(__file__).parent.parent


class Template:
    """Template class.
    Represents a template to replace entities.
    """

    def __init__(self, source: str, entities: List[Entity] = None):
        self.source = source
        self.entities = entities

    @property
    def entity_size(self):
        """Get the size of the entity."""
        return len(self.entities) if self.entities else 0

    def from_array(self, array: np.ndarray):
        """Create template from numpy array."""
        pass

    def to_array(self):
        """Convert to numpy array."""
        pass

    def from_text(self, text: str):
        """Create template from text."""
        pass

    def from_spacy_document(self, doc: spacy.tokens.Doc):
        """Create template from spacy document."""
        pass

    @property
    def entity_label(self) -> List[str]:
        """Get the class of the entities."""
        return [ent.label for ent in self.entities]

    @property
    def text(self) -> str:
        """Convert to text."""
        entities = [ent.text for ent in self.entities]
        return self.source.format(*entities)

    @classmethod
    def create_source(cls, text: str, entities: List[Entity]):
        """Create source from text and entities."""
        tokens: List[str] = text.split("\n")
        token_size = len(tokens)
        source = []
        prev_position = 0
        range_size = len(entities)
        for i in range(range_size):
            for j in range(prev_position, entities[i].start):
                source.append(tokens[j].split(" ")[0])
            source.append("{%d}" % i)
            prev_position = entities[i].end
            if i == range_size - 1 and entities[i].end < token_size:
                for j in range(entities[i].end, token_size):
                    source.append(tokens[j].split(" ")[0])
        return " ".join(source)

    @classmethod
    def from_iob_document(cls, text: str):
        """Create template from iob text."""
        entities = Entity.from_iob(text)
        source = cls.create_source(text, entities)
        return cls(source, entities)

    def to_iob(self) -> List[str]:
        """Convert to iob format."""
        tokens: List[str] = self.text.split(" ")
        iob_tokens: List[str] = ["O"] * len(tokens)
        for entity in self.entities:
            iob_tokens[entity.start] = "B-" + entity.label
            for i in range(entity.start + 1, entity.end):
                iob_tokens[i] = "I-" + entity.label

        return iob_tokens

    def mesh_entities(self, entities: Dict[str, List[Entity]]) -> List[List[Entity]]:
        """Generator to get several combinations of entities.
        Don't repeat samples.
        If there are repetead types of entities, then the entities of the same
        type won't be repeated.
        """
        # Get indices of entities for each class
        entity_indices = [range(len(entities[v])) for v in self.entity_label]
        sample_indices = product(*entity_indices)

        for indices in sample_indices:
            sample_to_remove: Dict[str, List[int]] = {}
            add_sample = True
            for index, label in enumerate(self.entity_label):
                if label in sample_to_remove:
                    if indices[index] in sample_to_remove[label]:
                        add_sample = False
                        break
                    # Add the index to the list of indices to remove
                    sample_to_remove[label].append(indices[index])
                else:
                    # Class is not in the sample to remove
                    sample_to_remove[label] = [indices[index]]
            if add_sample:
                sample = []
                for index, label in zip(indices, self.entity_label):
                    sample.append(entities[label][index])
                yield sample

    def generate_samples(
        self, entities: Dict[str, List[Entity]], n: int = 0
    ) -> Iterator["Template"]:
        """Generate samples from the template source.
        Update start and end span of the entities according to the template.
        """
        for sample_entities in self.mesh_entities(entities):
            displacement = 0
            for index, entity in enumerate(sample_entities):
                len_entity = len(entity)
                entity.start = self.entities[index].start + displacement
                entity.end = entity.start + len_entity
                displacement += len_entity - len(self.entities[index])
            yield self.__class__(self.source, sample_entities)


class TemplateCorpus:
    """Corpus class.
    Represents a corpus to identify entities.
    """

    def __init__(self, templates: List[Template]):
        self.templates = templates
        self._entities: Dict[str : List[Entity]] = {}
        self._is_clean_entities: Dict[str:bool] = {}
        self.add_entities_from_templates(templates)

    def add_entity(self, entity: Entity):
        """Add entity to the corpus.
        If type of entity is not in the corpus, add it.
        If type of entity is in the corpus, add entity to the list of entities
        and mark clean like False.
        """
        if entity.label not in self._entities:
            self._entities[entity.label] = [entity]
            self._is_clean_entities[entity.label] = True
        else:
            self._is_clean_entities[entity.label] = False
            self._entities[entity.label].append(entity)

    def add_entities_from_templates(self, templates: List[Template]):
        """Add entities from templates."""
        for template in templates:
            for entity in template.entities:
                self.add_entity(entity)

    def remove_duplicates(self, label: str):
        """Remove duplicates."""
        label_len = len(self._entities[label])
        index_remove = np.full(label_len, False)
        for i in range(label_len - 1):
            entity = self._entities[label][i]
            for j in range(i + 1, label_len):
                if entity == self._entities[label][j]:
                    index_remove[j] = True
                    break

        for i, is_remove in enumerate(index_remove):
            if not is_remove:
                yield self._entities[label][i]

    def get_entities(self, label: str):
        """Get all type entities.
        if entiies are not clean then remove duplicates.
        If type of entity is not in the corpus, return None.
        """
        if label not in self._entities:
            return None
        if not self._is_clean_entities[label]:
            # remove duplicates
            self._entities[label] = list(self.remove_duplicates(label))
            self._is_clean_entities[label] = True
        return self._entities[label]

    def from_text(self, text: str):
        """Create corpus from text."""
        pass

    def from_spacy_document(self, doc: spacy.tokens.Doc):
        """Create corpus from spacy document."""
        pass

    def from_spacy_documents(self, docs: List[spacy.tokens.Doc]):
        """Create corpus from spacy documents."""
        pass

    @classmethod
    def from_iob_file(cls, path: str, encoding: str = "utf-8", relative: bool = False):
        """Read file and create list of templates."""

        if relative:
            path = BASE_PATH / path
        with open(path, "r", encoding=encoding) as file:
            corpus_text = map(lambda x: x.strip(), file.read().split("\n\n"))

        templates: List[Template] = []
        for text in corpus_text:
            template = Template.from_iob_document(text)

            templates.append(template)
        return cls(templates)

    def from_iob_documents(self, docs: List[spacy.tokens.Doc]):
        """Create corpus from spacy documents."""
        pass

    def generate_samples(self, n: int = 0):
        """Generate samples from the corpus using templates.
        Use n samples from each template.
        Reemplace entities in the template with the entities from the corpus.
        """
        pass
