from typing import Optional

import nltk
import numpy as np
from nltk import word_tokenize

from src.scoring.method import MethodFactory, ConfidenceMethod


# Features for GUSUM algorithm
def sentence_position(location, tokenized_corpus):
    num_sentences = len(tokenized_corpus)
    if location == 0 or location + 1 == num_sentences:
        return 1.0
    return (num_sentences - location) / num_sentences


def sentence_length(sentence, tokenized_corpus):
    max_sentence_length = max(len(word_tokenize(sentence)) for sentence in tokenized_corpus)
    return len(word_tokenize(sentence)) / max_sentence_length


def proper_noun_ratio(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    proper_nouns = sum(1 for word, tag in tagged if tag == 'NNP')
    return proper_nouns / len(tokens) if tokens else 0


def numerical_token_ratio(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    numerical_tokens = sum(1 for word, tag in tagged if tag == 'CD')
    return 1 - (numerical_tokens / len(tokens)) if tokens else 0


def sentence_ranking(sentence, location, tokenized_corpus):
    value = sum([
        sentence_position(location, tokenized_corpus),
        sentence_length(sentence, tokenized_corpus),
        proper_noun_ratio(sentence),
        numerical_token_ratio(sentence)
    ])
    return value


@MethodFactory.register('gusum')
class GUSUM(ConfidenceMethod):
    def compute(self, text: str, claims: list[str], summary: Optional[str] = None) -> np.ndarray:
        sentence_ranks = np.zeros(len(claims))
        for i, sentence in enumerate(claims):
            value = sentence_ranking(sentence, i, claims)
            sentence_ranks[i] = value
        return sentence_ranks
