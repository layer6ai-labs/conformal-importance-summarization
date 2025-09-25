
<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" height="60"></a>
<a href="https://layer6.ai/"><img src="Signal1.jpg" height="60"></a>
</p>

# Document Summarization with Conformal Importance Guarantees 

This is the codebase accompanying the paper "Document Summarization with Conformal Importance Guarantees", published at NeurIPS 2025.

## Basic Usage

### Generating results
1. Create and activate a new virtual environment with python 3.11.
2. Install requirements (`pip install -r requirements.txt`).
3. Run `python main.py [args]` (run `python -m main.py --help` to see a list of options). Running with no arguments will automatically produce all results for our table 2 and figure 2 for all datasets. 

### Generating LLM Scores
If you want to download datasets and regenerate LLM scores with your own prompts and/or models rather than our pre-cached ones, the code can be found in `src/data/loaddata.py`. 
1. Delete or move the existing parquet files in `data/` which are read by default if present.
2. If accessing an external LLM provider, create a .env file in the project root directory containing API keys.
3. Call the corresponding `load_dataset_raw_xxx()` function which downloads and processes the data to create parquet files. To select which types of data to generate (e.g. labels, scores), uncomment the correesponding functions (e.g. addSBERTLabelColumn(), addLLMLabelColumn()) and add appropriate inputs. 

### Adding new scoring methods
To add a new importance scoring method:
1. Add the method's code to a new or existing file in src/scoring, subclassing ConfidenceMethod and using the `@MethodFactory.register("method-name")` decorator.
2. Update ./src/scoring/__init__.py with your new method.

### Running abstractive post-processing as an extra step after extractive Conformal Importance Summarization
To generate and evaluate the hybrid extractive/abstractive method: 
1. Ensure you have a .env file in the project root directory containing API keys for LLM providers. By default OpenAI and Gemini API keys are required.
2. Set your experiment hyperparameters in the `extr_abstr_experiments.py` file (lines 254-260)
3. run with `python extr_abstr_experiments.py`

### Running pure abstractive summarization baselines with LLMs
To generate and evaluate abstractive summary baselines: 
1. Ensure you have a .env file in the project root directory containing API keys for LLM providers. By default OpenAI and Gemini API keys are required.
2. Set your experiment hyperparameters in the `abstractive_experiment.py` file (lines 181-185)
3. run with `python abstractive_experiment.py`

## Organization
- data: Contains parquet files with pre-processed datasets containing sentence-tokenized texts, labels, and pre-computed importance scores for several LLMs. 
- src: 
    - data: 
        - dataset.py: classes for performing conformal summarization on entire datasets
        - instance.py: base class for summary texts
        - loaddata.py: Functions for creating and loading our datasets, including labels and language model scores
    - scoring: methods for computing importance scores for subclaims 
        ...
    - llm.py: various utilities for running score functions involving LLMs
    - evals.py: functions for evaluating the results of conformal summarization
        - ...
    - tasks
        - Subclasses for importance summarization
    - utils.py: utility functions
- main.py: Main method for running conformal importance summarization experiments (see above)
- extr_abstr_experiment.py: Main method for running abstractive post-processing on top of Conformal Importance Summarization (see above)
- abstractive_experiment.py: Main method for running pure abstractive summarizartion baseline experiments (see above)

# Citing

    @inproceedings{kuwahara2025,
        title={Document Summarization with Conformal Importance Guarantees}, 
        author={Bruce Kuwahara and Chen-Yuan Lin and Xiao Shi Huang and Kin Kwan Leung and Jullian Arta Yapeter and Ilya Stanevich and Felipe Perez and Jesse C. Cresswell},
        booktitle={Advances in Neural Information Processing Systems},
        year={2025},
    }

# License
This data and code is licensed under the MIT License, copyright by Layer 6 AI and Signal 1 AI.