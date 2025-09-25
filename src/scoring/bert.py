from typing import Optional

import numpy as np
from bert_score import score as bert_score

from src.scoring import ConfidenceMethod, MethodFactory


@MethodFactory.register('bertscore')
class BertScoreMethod(ConfidenceMethod):

    def __init__(self, lang: str = 'en'):
        self.lang = lang

    def compute(self, text: str, claims: list[str], summary: Optional[str] = None) -> np.ndarray:
        P, R, F1 = bert_score(claims, [summary], lang=self.lang, verbose=False)
        return F1.numpy()
