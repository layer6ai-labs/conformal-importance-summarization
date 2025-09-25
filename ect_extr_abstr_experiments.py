import json
import time
import re
from typing import Literal

import pandas as pd
import numpy as np
import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from google import genai

load_dotenv("./.envrc")


openai_client = OpenAI()
gemini_client = genai.Client()

MAX_RETRIES = 10


def llm_call(prompt_text: str, model: Literal["gpt-4o-mini", "gemini-2.5-flash"]):

    for attempt in range(MAX_RETRIES):
        try:
            if model == "gpt-4o-mini":
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                return response.choices[0].message.content
            if model == "gemini-2.5-flash":
                response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt_text,
                )
                return response.text
            else:
                raise ValueError(f"Model {model} not supported")
        except Exception as e:
            last_exception = e
            if attempt == MAX_RETRIES - 1:
                raise last_exception
            # Calculate delay with exponential backoff: base_delay * 2^attempt
            delay = 2 ** attempt
            print(f"Exception {type(e)}. Trying again in {delay}s")
            time.sleep(delay)
    # This should never be reached due to the raise in the loop
    raise RuntimeError


def load_cal_test(seed: int | None):
    cal_df = pd.read_parquet("../data/ECTSum/ECT_cal.parquet")
    test_df = pd.read_parquet("../data/ECTSum/ECT_test.parquet")

    if seed is not None:
        cal_size = len(cal_df)
        test_size = len(test_df)
        combined_df = pd.concat([cal_df, test_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        cal_df = combined_df.iloc[:cal_size].reset_index(drop=True)
        test_df = combined_df.iloc[cal_size:cal_size + test_size].reset_index(drop=True)
        assert len(cal_df) == cal_size
        assert len(test_df) == test_size

    return cal_df, test_df


def get_file_name(
    beta: float,
    alpha: float,
    seed: int | None,
    score_name: str,
    summary_model: str,
):
    seed_str = "" if seed is None else f"-seed-{seed}"
    return f"retain_ECT_experiment-summary-{summary_model}-beta-{round(beta, 2)}-alpha-{round(alpha, 2)}-{score_name}{seed_str}.json"


def run(
    beta: float,
    alpha: float,
    seed: int | None,
    score_name: str,
    summary_model: Literal["gpt-4o-mini", "gemini-2.5-flash"],
):
    out_file = get_file_name(beta, alpha, seed, score_name, summary_model)

    try:
        with open(out_file, "r") as f:
            res = [json.loads(l) for l in f.readlines()]
            lines_run = len(res)
            print(f"{lines_run} lines have already been run. Continuing from there.")
    except FileNotFoundError:
        lines_run = 0

    # shuffle the cal and test set together, then split again
    cal_df, test_df = load_cal_test(seed)

    # Count unique values to see if they are degenerate
    # unique_score_counts = pd.Series(np.concatenate(cal_df[score_name].values)).value_counts()
    # print(unique_score_counts.head(10))
    # they are highly degenerate, fix it!
    rng = np.random.RandomState(0)
    for df in [cal_df, test_df]:
        df[score_name] = df[score_name].apply(
            lambda scores: scores + 1e-6 * rng.randn(len(scores))
        )
    
    s_betas = []
    for _, row in cal_df.iterrows():
        label = row["input_sentences_labels"]
        score = row[score_name]
        argsort = np.argsort(score)[::-1]  # argsort the score from high to low.
        sorted_label = label[argsort]   # label w.r.t score from high to low
        sorted_score = score[argsort]   # score from high to low

        cumsum = np.cumsum(sorted_label)   # number of positive labels for the top N scores selected
        # cumsum/sum(label) = recall for top N scores selected. Output the first one. recall >= beta
        first_index = np.where(cumsum >= sum(label) * beta)[0][0]
        s_beta = sorted_score[first_index]
        s_betas.append(s_beta)

    threshold = np.percentile(s_betas, 100 * alpha * (len(cal_df) + 1) / len(cal_df))

    # validate that the percents >= beta is approximately 1-alpha
    for df in [cal_df, test_df]:
        percents = []
        for _, row in df.iterrows():
            label = row["input_sentences_labels"]
            num_positives = sum(label)
            score = row[score_name]
            selected = score >= threshold
            num_selected = sum(label[selected])
            percents.append(num_selected / num_positives)
        print((np.asarray(percents) >= beta).mean())

    responses = []
    with open(out_file, "a") as f:
        for i, (_, row) in tqdm.tqdm(enumerate(test_df.iterrows()), total=len(test_df)):
            if i < lines_run:
                # already ran
                continue
            label = row["input_sentences_labels"]
            positive_labels = label > 0.5
            score = row[score_name]
            selected = score >= threshold
            selected_positive = selected[positive_labels]
            selected_sentences = row["input_sentences"][selected]
            important_sentences = row["input_sentences"][positive_labels]
            input_text = " ".join(selected_sentences)
            prompt_text = (
                "Requirements:\n- Use more concise language to make the text shorter\n"
                "- Retain all of the information from the input text\n"
                "- Improve flow, coherence, and readability\n\n"
                f"Text to shorten, paraphrase and rewrite:\n{input_text}"
            )
            summary = llm_call(prompt_text, model=summary_model)
            responses.append(summary)

            retained_array = []
            for important_sentence in important_sentences:
                retain_prompt = (
                    "You will be given an important sentence from the original text and a generated "
                    "summary. Your goal is to determine whether the important sentence given is "
                    f"retained in the generated summary.\n\nImportant sentence:\n{important_sentence}\n\n"
                    f"Generated summary:\n{summary}\n\nOutput True if the important sentence is retained "
                    "in the generated summary. Output False otherwise."
                )
                retained_text = llm_call(retain_prompt, model="gpt-4o-mini").lower().strip(" .\"\'")
                if retained_text == "true":
                    retained = True
                elif retained_text == "false":
                    retained = False
                else:
                    print(f"RETAINED TEXT='{retained_text}'")
                    retained = None
                retained_array.append(retained)
            out_dict = {
                "selected_positive": selected_positive.tolist(),
                "retained_array": retained_array,
                "input_text": input_text,
                "summary": summary,
            }
            out_json = json.dumps(out_dict)
            f.write(out_json + "\n")
            print(len(input_text), len(responses[-1]), retained_array, selected_positive)


def analyze(
    beta: float,
    alpha: float,
    seed: int | None,
    score_name: str,
    summary_model: Literal["gpt-4o-mini", "gemini-2.5-flash"]
):
    in_file = get_file_name(beta, alpha, seed, score_name, summary_model)
    _, test_df = load_cal_test(seed)
    with open(in_file, "r") as f:
        res = [json.loads(l) for l in f.readlines()]
    df = pd.DataFrame(res)
    df["original_text"] = test_df["input"]
    df["selected_positive"] = df["selected_positive"].apply(np.asarray)
    df["retained_array"] = df["retained_array"].apply(np.asarray)
    df["selected_rate"] = df["selected_positive"].apply(np.mean)
    df["retained_rate"] = df["retained_array"].apply(np.mean)
    df["intersect"] = df.apply(lambda x: x["selected_positive"] & x["retained_array"], axis=1)
    df["retained_selected"] = df.apply(lambda x: x["intersect"].sum() / x["selected_positive"].sum(), axis=1)
    total_retained_rate = df["intersect"].apply(sum).sum() / df["selected_positive"].apply(sum).sum()
    total_sample_selected_rate = df["selected_positive"].apply(sum).sum() / df['selected_positive'].apply(len).sum()
    total_sample_retained_rate = df["retained_array"].apply(sum).sum() / df["retained_array"].apply(len).sum()
    word_count = lambda x: len(re.findall(r'\w+', x))
    res = {
        "average_sample_selected_rate": df["selected_rate"].mean(),
        "average_sample_retained_rate": df["retained_rate"].mean(),
        "recall_difference": df["selected_rate"].mean() - df["retained_rate"].mean(),
        "average_sample_retained_selected_rate": df["retained_selected"].mean(),
        "total_sample_selected_rate": total_sample_selected_rate,
        "total_sample_retained_rate": total_sample_retained_rate,
        "total_retained_selected_rate": total_retained_rate,
        "fraction_sample_full_retained": (df["retained_selected"] == 1).mean(),
        "average_fraction_length": (df["summary"].apply(len) / df["input_text"].apply(len)).mean(),
        "average_fraction_word": (df["summary"].apply(word_count) / df["input_text"].apply(word_count)).mean(),
        "average_fraction_total_length": (df["summary"].apply(len) / df["original_text"].apply(len)).mean(),
        "average_fraction_total_word": (df["summary"].apply(word_count) / df["original_text"].apply(word_count)).mean(),
        "coverage_before": (df["selected_rate"] >= beta).mean(),
        "coverage_after": (df["retained_rate"] >= beta).mean(),
        "coverage_difference": (df["selected_rate"] >= beta).mean() - (df["retained_rate"] >= beta).mean(),
    }
    print(pd.Series(res).to_csv(header=False))


if __name__ == '__main__':
    beta = 0.8
    alpha = 0.2
    seed = None  # can shuffle cal/test splits to get different CP results
    score_name = "gemini25_scores"
    summary_model = "gemini-2.5-flash"

    run(beta, alpha, seed, score_name, summary_model)
    analyze(beta, alpha, seed, score_name, summary_model)
