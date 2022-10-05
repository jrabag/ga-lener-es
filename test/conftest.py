import pytest


@pytest.fixture
def simple_iob_text() -> str:
    """Return simple iob text."""
    text = (
        "Sao B-LOC\n"
        "Paulo I-LOC\n"
        "( O\n"
        "Brasil B-LOC\n"
        ") O\n"
        ", O\n"
        "23 O\n"
        "may O\n"
        "( O\n"
        "EFECOM B-ORG\n"
        ") O\n"
        ". O"
    )
    return text


@pytest.fixture
def adjacent_iob_text() -> str:
    """Return adjacent iob text."""
    text = (
        "El O\n"
        "miembro O\n"
        "de O\n"
        "la O\n"
        "Comisión B-ORG\n"
        "Regional I-ORG\n"
        "de I-ORG\n"
        "UCE I-ORG\n"
        "Emilio B-PER\n"
        "Guerrero I-PER\n"
        "expondrá O\n"
        "la O\n"
        "posición O\n"
        "de O\n"
        "esta O\n"
        "organización O\n"
        "agraria O\n"
        "respecto O\n"
        "a O\n"
        "la O\n"
        "próxima O\n"
        "campaña O\n"
        "de O\n"
        "tomate O\n"
        ". O"
    )
    return text


@pytest.fixture
def special_iob_text() -> str:
    """Return special iob text."""
    text = (
        "Así O\n"
        "lo O\n"
        "aseguró O\n"
        "Yamil B-PER\n"
        "Chade I-PER\n"
        ", O\n"
        "asesor O\n"
        "del O\n"
        "púgil O\n"
        "borricua O\n"
        ", O\n"
        "quien O\n"
        "indicó O\n"
        "que O\n"
        "Camacho B-PER\n"
        "- O\n"
        "hijo O\n"
        "del O\n"
        "también O\n"
        "boxeador O\n"
        "puertorriqueño O\n"
        "Héctor B-PER\n"
        '"Macho" I-PER\n'
        "Camacho I-PER\n"
        "- O\n"
        "podría O\n"
        "combatir O\n"
        "en O\n"
        "las O\n"
        "categorías O\n"
        "de O\n"
        "130 O\n"
        "( O\n"
        "59 O\n"
        "kg. O\n"
        ") O\n"
        ", O\n"
        "135 O\n"
        "( O\n"
        "61 O\n"
        "kg. O\n"
        ") O\n"
        "y O\n"
        "140 O\n"
        "libras O\n"
        "( O\n"
        "63 O\n"
        "kg. O\n"
        ") O\n"
        "indistintamente O\n"
        "y O\n"
        "con O\n"
        "éxito O\n"
        ". O"
    )
    return text
