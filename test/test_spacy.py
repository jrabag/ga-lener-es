"""Test basic functions of the spacy."""

from typing import TYPE_CHECKING

import pytest
import spacy

if TYPE_CHECKING:
    from spacy.language import Language as SpacyLanguage
    from spacy.tokens import Doc as SpacyDoc


@pytest.fixture
def spacy_nlp() -> "SpacyLanguage":
    """Return a spacy nlp object."""
    return spacy.load("es_dep_news_trf")


# pylint: disable=redefined-outer-name
@pytest.fixture
def spacy_doc(spacy_nlp: "SpacyLanguage") -> "SpacyDoc":
    """Return a spacy doc object."""
    return spacy_nlp(
        (
            "Alberto Hernando Robles Restrepo celebró un contrato el 22 de marzo de "
            "2011 con el Banco BBVA"
        )
    )


class TestSpacy:
    """Test the feaure spacy"""

    def test_avaible_gpu(self):
        """Test the avaible gpu."""
        assert spacy.require_gpu()

    def test_spacy_tokenizer(self, spacy_doc):
        """Test the spacy tokenizer."""
        assert len(spacy_doc) == 17
