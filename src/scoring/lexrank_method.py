
# import lexrank
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from .method import ConfidenceMethod


class LexRankScore(ConfidenceMethod):

    def __init__(self, documents, threshold=0.1):
        self.lexrank = LexRank(documents, stopwords=STOPWORDS['en'])
        self.threshold = threshold
        self.name = "lexrank"

    def compute(self, text, claims, summary=None):
        scores = self.lexrank.rank_sentences(claims, threshold=self.threshold)
        return scores
