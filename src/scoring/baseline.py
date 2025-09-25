from typing import Optional

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.scoring.method import MethodFactory, ConfidenceMethod

__all__ = [
    'RandomMethod',
    'OrdinalMethod'
]


@MethodFactory.register('random')
class RandomMethod(ConfidenceMethod):

    def compute(self, text: str, claims: list[str], summary: Optional[str] = None) -> np.ndarray:
        return np.random.normal(size=len(claims))


@MethodFactory.register('ordinal')
class OrdinalMethod(ConfidenceMethod):
    def compute(self, text: str, claims: list[str], summary: Optional[str] = None) -> np.ndarray:
        scores = [0.0] * len(claims)
        for i, x in enumerate(range(1, len(claims) + 1)):
            scores[i] = len(claims) - x
        return np.asarray(scores)


@MethodFactory.register('rouge_l')
class RougeLMethod(ConfidenceMethod):
    def __init__(self, lang: str = 'en'):
        self.lang = lang
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def compute(self, text: str, claims: list[str], summary: Optional[str] = None) -> np.ndarray:
        if summary is None:
            raise ValueError("Summary is required for ROUGE-L scoring.")
        return np.asarray([
            self.scorer.score(claim, summary)['rougeL'].fmeasure for claim in claims
        ])


@MethodFactory.register('redundancy_penalty')
class RedundancyPenaltyMethod(ConfidenceMethod):
    def __init__(self, lang: str = 'en', model_name: str = 'all-MiniLM-L6-v2'):
        self.lang = lang
        self.model = SentenceTransformer(model_name)

    def compute(self, text: str, claims: list[str], summary: Optional[str] = None) -> np.ndarray:
        embeddings = self.model.encode(claims, normalize_embeddings=True)
        sim_matrix = cosine_similarity(embeddings)
        redundancy = sim_matrix.sum(axis=1) - 1  # remove self-similarity
        return 1 / (1 + redundancy)
