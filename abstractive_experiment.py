import json
import time
import os
import re
from typing import Literal

import numpy as np
import pandas as pd
import tqdm
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from src.data.loaddata import load_dataset

load_dotenv('./.env')
api_key_openai = os.getenv("OPENAI_API_KEY", "")
api_key_gemini = os.getenv("GEMINI_API_KEY", "")
print("Using OpenAI and/or Gemini API Keys")
openai_client = OpenAI(api_key=api_key_openai)
gemini_client = genai.Client(api_key=api_key_gemini)

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
            elif model == "gemini-2.5-flash":
                response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt_text,
                )
                return response.text
            else:
                raise ValueError(f"Model {model} not supported")
            break
        except Exception as e:
            last_exception = e
            if attempt == MAX_RETRIES - 1:
                raise last_exception
            # Calculate delay with exponential backoff: base_delay * 2^attempt
            delay = 2 ** attempt
            print(f"Exception {type(e)}. Trying again in {delay}s")
            time.sleep(delay)


def get_file_name(
    model: str,
    dataset: str,
    beta: float,
    num_icl_examples: int,
    seed: int | None,
):
    seed_str = "" if seed is None else f"-seed-{seed}"
    os.makedirs("experiments", exist_ok=True)
    return f"experiments/abstractive_experiment-model-{model}-dataset-{dataset}-beta-{round(beta, 2)}-num_icl_examples-{num_icl_examples}{seed_str}.json"

def run(model, dataset, beta, num_icl_examples, seed):
    out_file = get_file_name(model, dataset, beta, num_icl_examples, seed)
    cal_df, test_df = load_dataset(name=dataset, as_dataframe=True, use_pandas=True)

    if seed is not None:
        cal_size = len(cal_df)
        test_size = len(test_df)
        combined_df = pd.concat([cal_df, test_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        cal_df = combined_df.iloc[:cal_size].reset_index(drop=True)
        test_df = combined_df.iloc[cal_size : cal_size + test_size].reset_index(drop=True)

    examples_text = ""
    if num_icl_examples > 0:
        print(f"Loading {num_icl_examples} in-context examples")
        for i, (_, row) in tqdm.tqdm(
            enumerate(cal_df.head(num_icl_examples).iterrows()),
            total=num_icl_examples,
        ):
            examples_text += f"""Example {i}:
            Input text: {row["input"]}
            Important information to retain: {row["summary"]}
            \n"""

    with open(out_file, "x") as f:
        for i, (_, row) in tqdm.tqdm(
            enumerate(test_df.iterrows()), total=len(test_df)
        ):
            input_text = row["input"]
            important_sentences = row["summary_sentences"]
            ground_truth_summary = row["summary"]
            prompt_text = ""
            if num_icl_examples > 0:
                prompt_text += f"""
                Here are examples of what constitutes important information to include in the summary:

                {examples_text}
                """
            prompt_text += f"""
            Create an abstractive summary of the following text.

            Requirements:
            - Aim to retain {"at least " if beta < 1 else ""}{beta * 100}% of the important information
            - Use your own words and phrasing (abstractive, not extractive)

            Input text to summarize:
            {input_text}"""

            generated_summary = llm_call(prompt_text, model)

            retained_array = []
            for important_sentence in important_sentences:
                retain_prompt = (
                    "You will be given an important sentence from the original text and a generated "
                    "summary. Your goal is to determine whether the important sentence given is "
                    f"retained in the generated summary.\n\nImportant sentence:\n{important_sentence}\n\n"
                    f"Generated summary:\n{generated_summary}\n\nOutput True if the important sentence is retained "
                    "in the generated summary. Output False otherwise."
                )
                retained_text = llm_call(retain_prompt, model="gpt-4o-mini").lower().strip(" .\"'")
                if retained_text == "true":
                    retained = True
                elif retained_text == "false":
                    retained = False
                else:
                    print(f"RETAINED TEXT='{retained_text}'")
                    retained = None
                retained_array.append(retained)
            out_dict = {
                "retained_array": retained_array,
                "input_text": input_text,
                "generated_summary": generated_summary,
                "ground_truth_summary": ground_truth_summary,
            }
            out_json = json.dumps(out_dict)
            f.write(out_json + "\n")
            print(
                model,
                len(input_text),
                len(ground_truth_summary),
                len(generated_summary),
                retained_array,
            )


def analyze(model, dataset, beta, num_icl_examples, seed):
    in_file = get_file_name(model, dataset, beta, num_icl_examples, seed)

    with open(in_file, "r") as f:
        res = [json.loads(l) for l in f.readlines()]
    df = pd.DataFrame(res)
    df["retained_array"] = df["retained_array"].apply(np.asarray)
    df["retained_rate"] = df["retained_array"].apply(np.mean)
    total_sample_retained_rate = (
        df["retained_array"].apply(sum).sum()
        / df["retained_array"].apply(len).sum()
    )
    word_count = lambda x: len(re.findall(r"\w+", x))
    metrics = {
        "average_sample_retained_rate": df["retained_rate"].mean(),
        "total_sample_retained_rate": total_sample_retained_rate,
        "average_fraction_length": (
            df["generated_summary"].apply(len) / df["input_text"].apply(len)
        ).mean(),
        "average_fraction_word": (
            df["generated_summary"].apply(word_count)
            / df["input_text"].apply(word_count)
        ).mean(),
        "coverage": (df["retained_rate"] >= beta).mean(),
    }
    metrics_filename = in_file.replace(".json", ".csv")
    pd.Series(metrics).to_csv(metrics_filename, header=False)
    print(pd.Series(metrics))


if __name__ == "__main__":
    # model options provided by default: gpt-4o-mini, gemini-2.5-flash
    model = "gemini-2.5-flash"
    dataset = "ECTSum"
    beta = 0.8
    num_icl_examples = 10
    seed = None  # can shuffle cal/test splits to get different CP results
    run(model, dataset, beta, num_icl_examples, seed)
    analyze(model, dataset, beta, num_icl_examples, seed)
