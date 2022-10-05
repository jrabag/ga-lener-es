from typing import List

import pytest
from ga_ner.models import Entity
from ga_ner.templates import Template, TemplateCorpus


@pytest.fixture
def template_three_entities() -> Template:
    return Template(
        "{0} ( {1} ) , 23 may ( {2} ) .",
        [
            Entity("LOC", "Sao Paulo", 0, 2),
            Entity("LOC", "Brasil", 3, 4),
            Entity("ORG", "EFECOM", 9, 10),
        ],
    )


@pytest.fixture
def template_adjacent_entities():
    return Template(
        (
            "El miembro de la {0} {1} expondrá la posición "
            "de esta organización agraria respecto a la próxima campaña de tomate ."
        ),
        [
            Entity("ORG", "Comisión Regional de UCE", 4, 8),
            Entity("PER", "Emilio Guerrero", 8, 10),
        ],
    )


@pytest.fixture
def template_type(request, template_three_entities, template_adjacent_entities):
    """Fixture for testing templates.
    If parameter is 'three', return template_three_entities.
    If parameter is 'adjacent', return template_adjacent_entities.
    If parameter isn't matched return template_three_entities
    """
    if request.param == "adjacent":
        return template_adjacent_entities

    return template_three_entities


@pytest.fixture
def iob_list(request):
    """Fixture for testing templates.
    If parameter are 'three', 'adjacent'
    If parameter isn't matched return 'three'
    """
    if request.param == "adjacent":
        return ["O"] * 4 + ["B-ORG"] + ["I-ORG"] * 3 + ["B-PER", "I-PER"] + ["O"] * 15

    return ["B-LOC", "I-LOC", "O", "B-LOC"] + ["O"] * 5 + ["B-ORG"] + ["O"] * 2


class TestFromIOBText:
    def test_simple(self, simple_iob_text: str):
        """Test simple template."""

        template = Template.from_iob_document(simple_iob_text)
        assert template.entities == [
            Entity("LOC", "Sao Paulo", 0, 2),
            Entity("LOC", "Brasil", 3, 4),
            Entity("ORG", "EFECOM", 9, 10),
        ]

        assert template.source == ("{0} ( {1} ) , 23 may ( {2} ) .")

        assert template.text == ("Sao Paulo ( Brasil ) , 23 may ( EFECOM ) .")

    def test_adjacent_entities(self, adjacent_iob_text: str):

        template = Template.from_iob_document(adjacent_iob_text)
        assert template.entities == [
            Entity("ORG", "Comisión Regional de UCE", 4, 8),
            Entity("PER", "Emilio Guerrero", 8, 10),
        ]

        assert template.source == (
            "El miembro de la {0} {1} expondrá la posición "
            "de esta organización agraria respecto a la próxima campaña de tomate ."
        )

        assert template.text == (
            "El miembro de la Comisión Regional de UCE Emilio Guerrero expondrá la posición "
            "de esta organización agraria respecto a la próxima campaña de tomate ."
        )


class TestFromIOBFile:
    def test_count_templates(self):
        """Test count templates."""
        corpus = TemplateCorpus.from_iob_file(
            "test/data/iob_test.txt", encoding="iso-8859-1"
        )
        assert len(corpus.templates) == 4

    def test_get_entities(self):
        """Test get entities."""
        corpus = TemplateCorpus.from_iob_file(
            "test/data/iob_test.txt", encoding="iso-8859-1"
        )
        # Validate entity LOC
        entities = corpus.get_entities("LOC")
        assert len(entities) == 4
        assert entities[0].text == "Sao Paulo"
        assert entities[0].start == 0
        assert entities[0].end == 2
        assert entities[1].text == "Brasil"
        assert entities[1].start == 3
        assert entities[1].end == 4
        assert entities[2].text == "Cantabria"
        assert entities[2].start == 2
        assert entities[2].end == 3
        assert entities[3].text == "México"
        assert entities[3].start == 3
        assert entities[3].end == 4
        # Validate entity ORG
        entities = corpus.get_entities("ORG")
        assert len(entities) == 2
        assert entities[0].text == "EFE"
        assert entities[0].start == 9
        assert entities[0].end == 10
        assert entities[1].text == "Comisión Regional de UCE"
        assert entities[1].start == 4
        assert entities[1].end == 8
        # Validate entity PER
        entities = corpus.get_entities("PER")
        assert len(entities) == 1
        assert entities[0].text == "Emilio Guerrero"
        assert entities[0].start == 8
        assert entities[0].end == 10


class TestGenerateSamples:
    def test_simple_combination(self, template_three_entities: Template):
        """Test combination of entities."""
        entities = {
            "LOC": ["Sao Paulo", "Brasil", "Cantabria"],
            "ORG": ["EFECOM", "Comisión Regional de UCE"],
        }
        samples = template_three_entities.mesh_entities(entities)
        assert next(samples) == ["Sao Paulo", "Brasil", "EFECOM"]
        assert next(samples) == ["Sao Paulo", "Brasil", "Comisión Regional de UCE"]
        assert next(samples) == ["Sao Paulo", "Cantabria", "EFECOM"]
        assert next(samples) == ["Sao Paulo", "Cantabria", "Comisión Regional de UCE"]
        assert next(samples) == ["Brasil", "Sao Paulo", "EFECOM"]
        assert next(samples) == ["Brasil", "Sao Paulo", "Comisión Regional de UCE"]
        assert next(samples) == ["Brasil", "Cantabria", "EFECOM"]
        assert next(samples) == ["Brasil", "Cantabria", "Comisión Regional de UCE"]
        assert next(samples) == ["Cantabria", "Sao Paulo", "EFECOM"]
        assert next(samples) == ["Cantabria", "Sao Paulo", "Comisión Regional de UCE"]
        assert next(samples) == ["Cantabria", "Brasil", "EFECOM"]
        assert next(samples) == ["Cantabria", "Brasil", "Comisión Regional de UCE"]

    def test_simple_templates(self, template_three_entities: Template):
        """Test simple templates."""
        entities = {
            "LOC": [
                Entity("LOC", "Sao Paulo", 0, 2),
                Entity("LOC", "Brasil", 3, 4),
                Entity("LOC", "Cantabria", 2, 3),
            ],
            "ORG": [
                Entity("ORG", "EFECOM", 9, 10),
                Entity("ORG", "Comisión Regional de UCE", 4, 8),
            ],
        }
        samples = template_three_entities.generate_samples(entities)
        sample = next(samples)
        assert sample.text == "Sao Paulo ( Brasil ) , 23 may ( EFECOM ) ."
        assert sample.entities == [
            Entity("LOC", "Sao Paulo", 0, 2),
            Entity("LOC", "Brasil", 3, 4),
            Entity("ORG", "EFECOM", 9, 10),
        ]
        sample = next(samples)
        assert (
            sample.text
            == "Sao Paulo ( Brasil ) , 23 may ( Comisión Regional de UCE ) ."
        )
        assert sample.entities == [
            Entity("LOC", "Sao Paulo", 0, 2),
            Entity("LOC", "Brasil", 3, 4),
            Entity("ORG", "Comisión Regional de UCE", 9, 13),
        ]
        sample = next(samples)
        assert sample.text == "Sao Paulo ( Cantabria ) , 23 may ( EFECOM ) ."
        assert sample.entities == [
            Entity("LOC", "Sao Paulo", 0, 2),
            Entity("LOC", "Cantabria", 3, 4),
            Entity("ORG", "EFECOM", 9, 10),
        ]
        sample = next(samples)
        assert (
            sample.text
            == "Sao Paulo ( Cantabria ) , 23 may ( Comisión Regional de UCE ) ."
        )
        assert sample.entities == [
            Entity("LOC", "Sao Paulo", 0, 2),
            Entity("LOC", "Cantabria", 3, 4),
            Entity("ORG", "Comisión Regional de UCE", 9, 13),
        ]
        sample = next(samples)
        assert sample.text == "Brasil ( Sao Paulo ) , 23 may ( EFECOM ) ."
        assert sample.entities == [
            Entity("LOC", "Brasil", 0, 1),
            Entity("LOC", "Sao Paulo", 2, 4),
            Entity("ORG", "EFECOM", 9, 10),
        ]
        sample = next(samples)
        assert (
            sample.text
            == "Brasil ( Sao Paulo ) , 23 may ( Comisión Regional de UCE ) ."
        )
        assert sample.entities == [
            Entity("LOC", "Brasil", 0, 1),
            Entity("LOC", "Sao Paulo", 2, 4),
            Entity("ORG", "Comisión Regional de UCE", 9, 13),
        ]
        sample = next(samples)
        assert sample.text == "Brasil ( Cantabria ) , 23 may ( EFECOM ) ."
        assert sample.entities == [
            Entity("LOC", "Brasil", 0, 1),
            Entity("LOC", "Cantabria", 2, 3),
            Entity("ORG", "EFECOM", 8, 9),
        ]
        sample = next(samples)
        assert (
            sample.text
            == "Brasil ( Cantabria ) , 23 may ( Comisión Regional de UCE ) ."
        )
        assert sample.entities == [
            Entity("LOC", "Brasil", 0, 1),
            Entity("LOC", "Cantabria", 2, 3),
            Entity("ORG", "Comisión Regional de UCE", 8, 12),
        ]
        sample = next(samples)
        assert sample.text == "Cantabria ( Sao Paulo ) , 23 may ( EFECOM ) ."
        assert sample.entities == [
            Entity("LOC", "Cantabria", 0, 1),
            Entity("LOC", "Sao Paulo", 2, 4),
            Entity("ORG", "EFECOM", 9, 10),
        ]
        sample = next(samples)
        assert (
            sample.text
            == "Cantabria ( Sao Paulo ) , 23 may ( Comisión Regional de UCE ) ."
        )
        assert sample.entities == [
            Entity("LOC", "Cantabria", 0, 1),
            Entity("LOC", "Sao Paulo", 2, 4),
            Entity("ORG", "Comisión Regional de UCE", 9, 13),
        ]
        sample = next(samples)
        assert sample.text == "Cantabria ( Brasil ) , 23 may ( EFECOM ) ."
        assert sample.entities == [
            Entity("LOC", "Cantabria", 0, 1),
            Entity("LOC", "Brasil", 2, 3),
            Entity("ORG", "EFECOM", 8, 9),
        ]
        sample = next(samples)
        assert (
            sample.text
            == "Cantabria ( Brasil ) , 23 may ( Comisión Regional de UCE ) ."
        )
        assert sample.entities == [
            Entity("LOC", "Cantabria", 0, 1),
            Entity("LOC", "Brasil", 2, 3),
            Entity("ORG", "Comisión Regional de UCE", 8, 12),
        ]


@pytest.mark.parametrize(
    "template_type,iob_list",
    [("simple", "simple"), ("adjacent", "adjacent")],
    indirect=True,
)
def test_template_to_iob(template_type: Template, iob_list: List[str]):
    """Test template to IOB."""
    iob = template_type.to_iob()
    assert iob == iob_list
