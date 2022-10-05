"""File to test the models."""

from typing import List

import numpy as np
import pytest
from ga_ner.models import Document, Entity, Feature, Rule, Vocabulary


@pytest.fixture
def simple_arrange_feature() -> List[Feature]:
    """Simple arrange feature."""
    return [Feature(0, 2)]


@pytest.fixture
def simple_arrange_number_rule():
    """Return numpy array to init a Rule"""
    return np.array([1, 3, 1, 4])


class TestRule:
    """Test the rule class."""

    def test_init(self, simple_arrange_feature):
        """Test the init method."""
        rule = Rule(1, simple_arrange_feature, 0.0)
        assert rule.entity == 1
        assert rule.score == 0.0
        assert len(rule.tokens) == len(simple_arrange_feature)
        assert rule.tokens[0] == simple_arrange_feature[0]

    def test_init_from_list(self, simple_arrange_number_rule):
        """Test the from_number_array method."""
        rule = Rule.from_number_array(simple_arrange_number_rule)
        assert rule.entity == 1
        assert rule.score == 0.0
        assert len(rule.tokens) == 1
        assert rule.tokens[0] == Feature(3, 1)
        assert rule.mask == [True]

    def test_to_list(self, simple_arrange_feature: List[Feature]):
        """Test the to_list method."""
        rule = Rule(1, simple_arrange_feature, [False], 0.0)
        feature_list = simple_arrange_feature[0].to_list()
        assert rule.to_list() == [1] + feature_list + [int(False)] + [0.0]

    def test_to_vector(self, simple_arrange_feature: List[Feature]):
        """Test the to_vector method."""
        rule = Rule(1, simple_arrange_feature, [False], 0.0)
        feature_list = simple_arrange_feature[0].to_list()
        assert rule.to_vector().tolist() == [1.0] + feature_list + [int(False)] + [0.0]


class TestDocument:
    """Test the document class."""

    def test_from_iob_simple(self, simple_iob_text: str):
        """Test create document from iob text."""
        document = Document.from_iob(simple_iob_text)
        assert document.entities == [
            Entity("LOC", "Sao Paulo", 0, 2),
            Entity("LOC", "Brasil", 3, 4),
            Entity("ORG", "EFECOM", 9, 10),
        ]
        assert len(document.tokens) == len(simple_iob_text.split("\n"))

    def test_from_iob_adjacent(self, adjacent_iob_text: str):
        """Test create document from iob text. with adjacent entities."""

        document = Document.from_iob(adjacent_iob_text)
        assert document.entities == [
            Entity("ORG", "Comisión Regional de UCE", 4, 8),
            Entity("PER", "Emilio Guerrero", 8, 10),
        ]
        assert len(document.tokens) == len(adjacent_iob_text.split("\n"))

    def test_from_iob_simple_iob_with_iob_vocab(self, simple_iob_text: str):
        """Test create document from iob text. with adjacent entities."""
        enconding = "iso-8859-1"
        vocab_conll = Vocabulary.from_iob("data/train/esp.train.txt", enconding, 10)

        document = Document.from_iob(simple_iob_text, vocab=vocab_conll)

        assert document.tokens[0] == [Feature(0, 13), Feature(1, 37), Feature(2, 1863)]
        assert document.tokens[1] == [Feature(0, 13), Feature(1, 23), Feature(2, 1864)]
        assert document.tokens[2] == [Feature(0, 14), Feature(1, 35), Feature(2, 1)]
        assert document.tokens[3] == [Feature(0, 13), Feature(1, 23), Feature(2, 1865)]
        assert document.tokens[4] == [Feature(0, 14), Feature(1, 35), Feature(2, 3)]
        assert document.tokens[5] == [Feature(0, 14), Feature(1, 35), Feature(2, 4)]
        assert document.tokens[6] == [Feature(0, 10), Feature(1, 30), Feature(2, 1441)]
        assert document.tokens[7] == [Feature(0, 9), Feature(1, 23), Feature(2, 6)]
        assert document.tokens[8] == [Feature(0, 14), Feature(1, 35), Feature(2, 1)]
        assert document.tokens[9] == [Feature(0, 13), Feature(1, 7), Feature(2, 195)]
        assert document.tokens[10] == [Feature(0, 14), Feature(1, 35), Feature(2, 3)]
        assert document.tokens[11] == [Feature(0, 14), Feature(1, 35), Feature(2, 8)]

    def test_from_iob_simple_with_vocab_bert(self, simple_iob_text: str):
        """Test create document from text with simple entities."""
        # enconding = "iso-8859-1"
        vocab_conll = Vocabulary.from_file(
            "data/vocabs/bert-base-uncased-vocab.txt", include_special=False
        )

        document = Document.from_iob(simple_iob_text, vocab=vocab_conll)
        assert document.tokens[0] == [Feature(0, 13), Feature(1, 37), Feature(2, 101)]
        assert document.tokens[1] == [Feature(0, 13), Feature(1, 23), Feature(2, 101)]
        assert document.tokens[2] == [Feature(0, 14), Feature(1, 35), Feature(2, 1007)]
        assert document.tokens[3] == [Feature(0, 13), Feature(1, 23), Feature(2, 101)]
        assert document.tokens[4] == [Feature(0, 14), Feature(1, 35), Feature(2, 1008)]
        assert document.tokens[5] == [Feature(0, 14), Feature(1, 35), Feature(2, 1011)]
        assert document.tokens[6] == [Feature(0, 10), Feature(1, 30), Feature(2, 2604)]
        assert document.tokens[7] == [Feature(0, 9), Feature(1, 23), Feature(2, 2090)]
        assert document.tokens[8] == [Feature(0, 14), Feature(1, 35), Feature(2, 1007)]
        assert document.tokens[9] == [Feature(0, 13), Feature(1, 7), Feature(2, 101)]
        assert document.tokens[10] == [Feature(0, 14), Feature(1, 35), Feature(2, 1008)]
        assert document.tokens[11] == [Feature(0, 14), Feature(1, 35), Feature(2, 1013)]

    def test_from_iob_simple_with_vocab_distibert(self, simple_iob_text: str):
        """Test create document from text with simple entities."""
        # enconding = "iso-8859-1"
        vocab = Vocabulary.from_file(
            "data/vocabs/distilbert-base-uncased-vocab.txt", include_special=False
        )

        document = Document.from_iob(simple_iob_text, vocab=vocab)

        assert document.tokens[0] == [Feature(0, 13), Feature(1, 37), Feature(2, 4)]
        assert document.tokens[1] == [Feature(0, 13), Feature(1, 23), Feature(2, 16036)]
        assert document.tokens[2] == [Feature(0, 14), Feature(1, 35), Feature(2, 1148)]
        assert document.tokens[3] == [Feature(0, 13), Feature(1, 23), Feature(2, 5831)]
        assert document.tokens[4] == [Feature(0, 14), Feature(1, 35), Feature(2, 1136)]
        assert document.tokens[5] == [Feature(0, 14), Feature(1, 35), Feature(2, 1018)]
        assert document.tokens[6] == [Feature(0, 10), Feature(1, 30), Feature(2, 2706)]
        assert document.tokens[7] == [Feature(0, 9), Feature(1, 23), Feature(2, 1583)]
        assert document.tokens[8] == [Feature(0, 14), Feature(1, 35), Feature(2, 1148)]
        assert document.tokens[9] == [Feature(0, 13), Feature(1, 7), Feature(2, 4)]
        assert document.tokens[10] == [Feature(0, 14), Feature(1, 35), Feature(2, 1136)]
        assert document.tokens[11] == [Feature(0, 14), Feature(1, 35), Feature(2, 1010)]

    def test_to_array(self):
        """Test to transform Documento to array."""
        tokens = [
            [Feature(0, 13), Feature(1, 37), Feature(2, 1863)],
            [Feature(0, 13), Feature(1, 23), Feature(2, 1864)],
            [Feature(0, 14), Feature(1, 35), Feature(2, 1)],
            [Feature(0, 13), Feature(1, 23), Feature(2, 1865)],
            [Feature(0, 14), Feature(1, 35), Feature(2, 3)],
            [Feature(0, 14), Feature(1, 35), Feature(2, 4)],
            [Feature(0, 10), Feature(1, 30), Feature(2, 1441)],
            [Feature(0, 9), Feature(1, 23), Feature(2, 6)],
            [Feature(0, 14), Feature(1, 35), Feature(2, 1)],
            [Feature(0, 13), Feature(1, 7), Feature(2, 195)],
            [Feature(0, 14), Feature(1, 35), Feature(2, 3)],
            [Feature(0, 14), Feature(1, 35), Feature(2, 8)],
        ]

        entities = [
            Entity("LOC", "Sao Paulo", 0, 2),
            Entity("LOC", "Brasil", 3, 4),
            Entity("ORG", "EFECOM", 9, 10),
        ]

        vocab_ent = Vocabulary.from_list(["LOC", "ORG"], include_special=False)

        document = Document(tokens, entities, vocab_ent=vocab_ent)
        document_array: np.ndarray = document.to_array()
        assert document_array.shape == (12 * 4,)
        # Token 0
        assert document_array[0:4].tolist() == [13, 37, 1863, 1]
        # Token 1
        assert document_array[4:8].tolist() == [13, 23, 1864, 1]
        # Token 2
        assert document_array[8:12].tolist() == [14, 35, 1, 0]
        # Token 3
        assert document_array[12:16].tolist() == [13, 23, 1865, 1]
        # Token 4
        assert document_array[16:20].tolist() == [14, 35, 3, 0]
        # Token 5
        assert document_array[20:24].tolist() == [14, 35, 4, 0]
        # Token 6
        assert document_array[24:28].tolist() == [10, 30, 1441, 0]
        # Token 7
        assert document_array[28:32].tolist() == [9, 23, 6, 0]
        # Token 8
        assert document_array[32:36].tolist() == [14, 35, 1, 0]
        # Token 9
        assert document_array[36:40].tolist() == [13, 7, 195, 2]
        # Token 10
        assert document_array[40:44].tolist() == [14, 35, 3, 0]
        # Token 11
        assert document_array[44:48].tolist() == [14, 35, 8, 0]

    def test_sampling_with_iob_vocab(self):
        """Test sampling with IOB vocab.
        Document without special tokens
        """
        vocab_conll = Vocabulary.from_file(
            "data/vocabs/conn-2002-uncased-vocab.txt", include_special=False
        )

        tokens = [
            [Feature(0, 13), Feature(1, 37), Feature(2, 1863)],
            [Feature(0, 13), Feature(1, 23), Feature(2, 1864)],
            [Feature(0, 14), Feature(1, 35), Feature(2, 1)],
        ]
        document = Document(tokens, [], vocab=vocab_conll)
        sampling = list(document.sampling(3))
        assert len(sampling) == (3**2 + 3**3 + 3**2)
        assert sampling[0] == (2782, 13, 13)
        assert sampling[1] == (2782, 13, 42)
        assert sampling[2] == (2782, 13, 1945)
        assert sampling[-3] == (1945, 14, 2783)
        assert sampling[-2] == (1945, 54, 2783)
        assert sampling[-1] == (1945, 82, 2783)

        document = Document(tokens, [], vocab=vocab_conll)
        sampling = list(document.sampling(4))
        assert len(sampling) == (3**3 + 3**3)

        document = Document(tokens, [], vocab=vocab_conll)
        sampling = list(document.sampling(5))
        assert len(sampling) == (3**3)
