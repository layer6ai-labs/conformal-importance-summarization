from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sentence_transformers import SentenceTransformer

from src.scoring.method import MethodFactory, ConfidenceMethod


# Custom centrality measure from Gong et al. (2022)
@MethodFactory.register('sentence-centrality')
class SentenceCentralityMethod(ConfidenceMethod):
    def __init__(self, model: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model)

    def compute(self, text: str, claims: list[str], summary: Optional[str] = None) -> np.ndarray:
        embeddings = self.model.encode(claims)
        distance_matrix = squareform(pdist(embeddings, metric='cosine'))
        similarity_matrix = 1 - distance_matrix

        n = similarity_matrix.shape[0]
        if n == 1:
            return np.array([1.0])
        sums = np.cumsum(similarity_matrix[:, ::-1], axis=1)
        SC = [sums[i, n - i - 2] / (n - i - 1) for i in range(n - 1)]
        SC.append(np.mean(SC))
        SC = np.array(SC)
        denominator = (SC.max() - SC.min())
        if denominator == 0:
            return SC
        return (SC - SC.min()) / denominator
