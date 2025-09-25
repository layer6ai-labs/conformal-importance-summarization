# conformal-llm

## Basic Usage

### Generating results
1. Create and activate a new virtual environment
2. Install requirements (`pip install -r requirements.txt`)
3. Run `python main.py [args]` (run `python -m main.py --help` to see a list of options). Running with no arguments will automatically produce all results for our table 2 and figure 2 for all datasets. 

### Generating LLM Scores
If you want to downlaod datasets and regenerate LLM scores with your own prompts and/or models rather than our pre-cached ones, the code can be found in src/data/loaddata.py. 
1. Delete or move the existing parquet files (they're read by default if present)
2. Call the corresponding load_dataset_raw_xxx() function
3. calls download->read->processing (add the "input"/"summary"/"sentences" columns)-add columns pipeline in sequence.
The labels and scores are generated after processing, either via addSBERTLabelColumn() or addLLMLabelColumn(). 

### Adding new scoring methods
In order to add a new importance method:
1. Add the method's code to a new or existing file in src/scoring, subclassing ConfidenceMethod and using the `@MethodFactory.register("method-name")` decorator
2. Update ./src/scoring/__init__.py with your new method

## Organization
- data: Contains parquet files with pre-processed datasets containing sentence-tokenized texts, labels, and our LLM-based scores. 
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
- main.py: Main method for running experiments (see above)


