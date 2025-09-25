import string
import numpy as np

from sentence_transformers import SentenceTransformer, util
from rouge import Rouge

from tqdm import tqdm
tqdm.pandas()

rouge = Rouge()
sbert = SentenceTransformer('all-mpnet-base-v2')

def compute_score(candidate_summary, reference_summary, score_fn):
    if isinstance(reference_summary, (list, np.ndarray)):
        scores = [score_fn(candidate_summary, ref_sentence) for ref_sentence in reference_summary]
        return max(scores) if scores else 0.0
    return score_fn(candidate_summary, reference_summary)


def greedy_extractive_labels_generic(article_sentences, reference_summary, reference_summary_sentences, score_fn,
                                     improvement_threshold=0.01):
    sentence_scores = [(idx, compute_score(sentence, reference_summary_sentences, score_fn))
                       for idx, sentence in enumerate(article_sentences)]
    sentence_scores.sort(key=lambda x: x[1], reverse=True)

    selected_indices = []
    current_summary = ""
    current_score = 0.0

    for idx, _ in sentence_scores:
        candidate_summary = (current_summary + " " + article_sentences[idx]).strip() if current_summary else \
            article_sentences[idx]
        candidate_score = compute_score(candidate_summary, reference_summary, score_fn)
        improvement = candidate_score - current_score

        if improvement > improvement_threshold:
            selected_indices.append(idx)
            current_summary = candidate_summary
            current_score = candidate_score

    labels = [1 if i in selected_indices else 0 for i in range(len(article_sentences))]
    return labels


def sbert_score(candidate_summary, reference_summary, model=sbert):
    if model is None:
        model = SentenceTransformer('all-mpnet-base-v2')
    cand_emb = model.encode(candidate_summary, convert_to_tensor=True)
    ref_emb = model.encode(reference_summary, convert_to_tensor=True)
    return util.pytorch_cos_sim(cand_emb, ref_emb).item()


def is_effectively_empty(text):
    stripped = text.strip().translate(str.maketrans('', '', string.punctuation))
    return len(stripped) == 0


def compute_rouge_scores(candidate_summary, reference_summary, rouge=rouge):
    if is_effectively_empty(candidate_summary):
        return None
    return rouge.get_scores(candidate_summary, reference_summary)[0]


def rouge1_score(candidate_summary, reference_summary, rouge=rouge):
    scores = compute_rouge_scores(candidate_summary, reference_summary, rouge)
    return 0.0 if scores is None else scores['rouge-1']['f']


def rouge2_score(candidate_summary, reference_summary, rouge=rouge):
    scores = compute_rouge_scores(candidate_summary, reference_summary, rouge)
    return 0.0 if scores is None else scores['rouge-2']['f']


def rougel_score(candidate_summary, reference_summary, rouge=rouge):
    scores = compute_rouge_scores(candidate_summary, reference_summary, rouge)
    return 0.0 if scores is None else scores['rouge-l']['f']


def addSBERTLabelColumn(df):
    df['input_sentences_labels'] = df.progress_apply(
        lambda row: greedy_extractive_labels_generic(row['input_sentences'], row['summary'], row['summary_sentences'],
                                                     sbert_score, improvement_threshold=0.05), axis=1)
    return df
