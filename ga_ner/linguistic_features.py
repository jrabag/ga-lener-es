"""Enums to define linguistic features."""
from typing import Dict

# Define Universal Dependency Relations
# see more: https://universaldependencies.org/u/dep/
dep_tags: Dict[str, int] = {
    "": 0,
    "UNK": 1,
    "EOL": 2,
    "acl": 3,  # clausal modifier of noun (adnominal clause)
    "advcl": 4,  # adverbial clause modifier
    "advmod": 5,  # adverbial modifier
    "amod": 6,  # adnominal modifier
    "appos": 7,  # appositional modifier
    "aux": 8,  # auxiliary
    "case": 9,  # case marking
    "cc": 10,  # coordinating conjunction
    "ccomp": 11,  # clausal complement
    "clf": 12,  # classifier
    "compound": 13,  # compound
    "conj": 14,  # conjunct
    "cop": 15,  # copula
    "csubj": 16,  # clausal subject
    "dep": 17,  # unspecified dependency
    "det": 18,  # determiner
    "discourse": 19,  # discourse element
    "dislocated": 20,  # dislocated elements
    "expl": 21,  # explanatory element
    "fixed": 22,  # fixed multiword token
    "flat": 23,  # flat multiword token
    "goeswith": 24,  # goes with
    "iobj": 25,  # indirect object
    "list": 26,  # list marker
    "mark": 27,  # marker
    "nmod": 28,  # nominal modifier
    "nsubj": 29,  # nominal subject
    "nummod": 30,  # numeric modifier
    "obj": 31,  # object
    "obl": 32,  # oblique nominal
    "orphan": 33,  # orphan
    "parataxis": 34,  # parataxis
    "punct": 35,  # punctuation
    "reparandum": 36,  # overridden disfluency
    "ROOT": 37,  # root
    "vocative": 38,  # vocative
    "xcomp": 39,  # open clausal complement
    "advmod:emph": 40,  # emphasizing word, intensifier
    "advmod:lmod": 41,  # locative adverbial modifier
    "aux:pass": 42,  # passive auxiliary
    "compound:lvc": 43,  # light verb construction
    "compound:prt": 44,  # phrasal verb particle
    "compound:redep": 45,  # reduplicated compounds
    "compound:svc": 46,  # serial verb compounds
    "csubj:pass": 47,  # clausal passive subject
    "det:numgov": 48,  # pronominal quantifier governing the case of the noun
    "det:nummod": 49,  # pronominal quantifier agreeing in case with the noun
    "det:possp": 50,  # possessive determiner
    "expl:impers": 51,  # impersonal explicative
    "expl:pass": 52,  # reflexive pronoun used in reflexive passive
    "expl:pv": 53,  # reflexive clitic with an inherently reflexive verb
    "nmod:poss": 54,  # possessive nominal modifier
    "nmod:tmod": 55,  # temporal modifier
    "nsubj:pass": 56,  # passive nominal subject
    "nummod:gov": 57,  # numeric modifier governing the case of the noun
    "obl:agent": 58,  # agent modifier
    "obl:arg": 59,  # oblique argument
    "obl:lmod": 60,  # locative modifier
    "obl:tmod": 61,  # temporal modifier
}

# Define Universal POS tags
# https://universaldependencies.org/u/pos/
pos_tags: Dict[str, int] = {
    "": 0,
    "UNK": 1,
    "ADJ": 2,  # adjective
    "ADP": 3,  # adposition
    "ADV": 4,  # adverb
    "AUX": 5,  # auxiliary
    "CCONJ": 6,  # coordinating conjunction
    "DET": 7,  # determiner
    "INTJ": 8,  # interjection
    "NOUN": 9,  # noun
    "NUM": 10,  # numeral
    "PART": 11,  # particle
    "PRON": 12,  # pronoun
    "PROPN": 13,  # proper noun
    "PUNCT": 14,  # punctuation
    "SCONJ": 15,  # subordinating conjunction
    "SYM": 16,  # symbol
    "VERB": 17,  # verb
    "X": 18,  # other
}
