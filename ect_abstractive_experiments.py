import json
import os
import re
import time

import numpy as np
import pandas as pd
import tqdm
from dotenv import load_dotenv
from google import genai
from openai import OpenAI

load_dotenv('./.env')
api_key_openai = os.getenv("OPENAI_API_KEY", "")
api_key_gemini = os.getenv("GEMINI_API_KEY", "")
print(api_key_openai)
print(api_key_gemini)
openai_client = OpenAI(api_key=api_key_openai)
gemini_client = genai.Client(api_key=api_key_gemini)


def llm_call(prompt_text, model="gpt-4o-mini"):
    max_retries = 10
    for attempt in range(max_retries):
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
            if attempt == max_retries - 1:
                raise last_exception
            # Calculate delay with exponential backoff: base_delay * 3^attempt
            delay = 3**attempt
            print(f"Retrying... {attempt + 1} of {max_retries}")
            time.sleep(delay)


def run(beta, num_examples, seed, model):
    cal_df = pd.read_parquet("../data/ECTSum/ECT_cal.parquet")
    test_df = pd.read_parquet("../data/ECTSum/ECT_test.parquet")
    out_file = f"abstractive_ECT_experiment-beta-{round(beta, 2)}-num_examples-{num_examples}-{model}.json"

    # shuffle the cal and test set together, then split again
    if seed is not None:
        cal_size = len(cal_df)
        test_size = len(test_df)
        combined_df = pd.concat([cal_df, test_df], ignore_index=True)
        combined_df = combined_df.sample(
            frac=1, random_state=seed
        ).reset_index(drop=True)
        cal_df = combined_df.iloc[:cal_size].reset_index(drop=True)
        test_df = combined_df.iloc[
            cal_size : cal_size + test_size
        ].reset_index(drop=True)
        assert len(cal_df) == cal_size
        assert len(test_df) == test_size

    examples_text = ""
    if num_examples > 0:
        print(f"Loading {num_examples} in-context examples")
        for i, (_, row) in tqdm.tqdm(
            enumerate(cal_df.head(num_examples).iterrows()),
            total=num_examples,
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
            if num_examples > 0:
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
                retained_text = llm_call(retain_prompt).lower().strip(" .\"'")
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


def analyze(beta, num_examples, model):
    in_file = f"abstractive_ECT_experiment-beta-{round(beta, 2)}-num_examples-{num_examples}-{model}.json"

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
    res = {
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
    pd.Series(res).to_csv(
        f"abstractive_ECT_experiment-beta-{round(beta, 2)}-num_examples-{num_examples}-{model}.csv"
    )
    print(pd.Series(res))


if __name__ == "__main__":
    model = "gemini-2.5-flash"
    beta = 0.9
    num_examples = 10
    seed = None  # can shuffle cal/test splits to get different CP results
    run(beta, num_examples, seed, model)
    analyze(beta, num_examples, model)
