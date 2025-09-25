from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from src.scoring.method import MethodFactory, ConfidenceMethod
from .graph_methods import similarity_matrix_from_sbert


@MethodFactory.register("cosine-centrality")
class CosineCentralityMethod(ConfidenceMethod):

    def __init__(self):
        print("Initializing sentence transformer...")
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    def compute(self, text: str, claims: list[str], summary: Optional[str] = None) -> np.ndarray:
        distance_matrix = similarity_matrix_from_sbert(self.sentence_transformer, claims)
        centralities = distance_matrix.sum(axis=1)
        return centralities
