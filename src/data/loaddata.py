from .download_utils import *
from .dataset_loaders import *
from .processing_utils import *
from .scoring_utils import *
from .llm_utils import *

import os
from pathlib import Path
from tqdm import tqdm
from src.data.dataset import ImportanceDataset
import polars as pl
import pandas as pd

# Enable tqdm progress bar for pandas
tqdm.pandas()

DATA_PATH = Path(__file__).resolve().parents[2] / "data"

randomseed = 3


def load_dataset_raw_ect(path, filename_cal, filename_test):
    downloadECTrawdata(target_directory=path)
    df = loadECTrawdata(path)
    df = processECTRawData(df)
    print("total {} samples".format(df.shape[0]))
    # df = addSBERTLabelColumn(df)
    # df = addLLMLabelColumn(df, LM=lm_gpt4omini)
    df = addLLMScoreColumns(df)
    df = df.reset_index(drop=True)
    df_cal = df.sample(n=100, random_state=randomseed)
    df_test = df.drop(df_cal.index)
    df_cal.to_parquet('{}'.format(filename_cal), index=False)
    df_test.to_parquet('{}'.format(filename_test), index=False)
    return df_cal, df_test


def load_dataset_raw_CSDS(path, filename_cal, filename_test):
    downloadCSDSrawdata(target_directory=path)
    df = loadCSDSrawdata(path)
    # df = addSBERTLabelColumn(df)
    # df = addLLMLabelColumn(df, LM=lm_gpt4omini)
    df = addLLMScoreColumns(df)
    df = df.reset_index(drop=True)
    df_cal = df.sample(n=100, random_state=randomseed)
    df_test = df.drop(df_cal.index)
    df_cal.to_parquet('{}'.format(filename_cal), index=False)
    df_test.to_parquet('{}'.format(filename_test), index=False)
    return df_cal, df_test


def load_dataset_raw_tldr(path, filename_cal, filename_test):
    downloadTLDRrawdata(target_directory=path)
    df = loadTLDRrawdata(path)
    df = processTLDRRawData(df)
    df = addSBERTLabelColumn(df)
    # df = addLLMLabelColumn(df, LM=lm_gpt4omini)
    df = addLLMScoreColumns(df)
    df = df.reset_index(drop=True)
    df_cal = df.sample(n=100, random_state=randomseed)
    df_test = df.drop(df_cal.index)
    df_cal.to_parquet('{}'.format(filename_cal), index=False)
    df_test.to_parquet('{}'.format(filename_test), index=False)
    return df_cal, df_test


def load_dataset_raw_MTS(path, filename_cal, filename_test):
    downloadMTSrawdata(target_directory=path)
    df = loadMTSrawdata(path)
    df = processMTSRawData(df)
    # df = addSBERTLabelColumn(df)
    df = addLLMLabelColumn(df, LM=lm_gpt4omini)
    df = addLLMScoreColumns(df)
    df = df.reset_index(drop=True)
    df_cal = df.sample(n=100, random_state=randomseed)
    df_test = df.drop(df_cal.index)
    df_cal.to_parquet('{}'.format(filename_cal), index=False)
    df_test.to_parquet('{}'.format(filename_test), index=False)
    return df_cal, df_test


def load_dataset(name="ECTSum", label_col="input_sentences_labels", as_dataframe=False, use_pandas=False):
    """
    Loading datasets, return a polars or pandas dataframe.
    Dataset is selected by the name input option:
    - nothing: the default dataset
    - MTS: the MTS_dialogue dataset, for healthcare
    - ECT: the ECTSum dataset, for finance

    If MTS_dialogue dataset, containing ["input", "summary", "perspective"].
    - Input: an array of strings. Each string contains the Q&A between Dcotor and Patient
    - Summary: the corresponding section in the report
    If MTS_dialogue dataset, containing ["input", "summary"].
    - Input: an array of strings. Each string contains one sentence from the call recording
    - Summary: the summary of the call recording transcription

    """
    if name == "tldr":
        filename_cal = DATA_PATH / 'TLDR/TLDR_cal.parquet'
        filename_test = DATA_PATH / 'TLDR/TLDR_test.parquet'
    elif name == 'tldr_fulltext':
        filename_cal = DATA_PATH / 'TLDR_full/TLDRfull_cal.parquet'
        filename_test = DATA_PATH / 'TLDR_full/TLDRfull_test.parquet'
    elif name == "MTS":
        filename_cal = DATA_PATH / 'MTS/MTS_cal.parquet'
        filename_test = DATA_PATH / 'MTS/MTS_test.parquet'
    elif name == "ECTSum":
        filename_cal = DATA_PATH / 'ECTSum/ECT_cal.parquet'
        filename_test = DATA_PATH / 'ECTSum/ECT_test.parquet'
    elif name == "CSDS":
        filename_cal = DATA_PATH / 'CSDS/CSDS_cal.parquet'
        filename_test = DATA_PATH / 'CSDS/CSDS_test.parquet'
    elif name == "CNNDM":
        filename_cal = DATA_PATH / 'CNNDailyMail/CNNDM_cal.parquet'
        filename_test = DATA_PATH / 'CNNDailyMail/CNNDM_test.parquet'
    else:
        raise ValueError(f"Dataset {name} not found, check the name to be within [tldr, tldr_fulltext, MTS, ECTSum, CSDS, CNNDM]")

    print(filename_cal)
    print(filename_test)
    if os.path.exists(filename_cal) and os.path.exists(filename_test):
        if use_pandas:
            df_cal = pd.read_parquet(filename_cal)
            df_test = pd.read_parquet(filename_test)
        else:
            df_cal = pl.read_parquet(filename_cal)
            df_test = pl.read_parquet(filename_test)
    else:
        if name == "tldr":
            directory = DATA_PATH / 'TLDR/'
            df_cal, df_test = load_dataset_raw_tldr(directory, filename_cal, filename_test)
        elif name == "MTS":
            directory = DATA_PATH / 'MTS/'
            df_cal, df_test = load_dataset_raw_MTS(directory, filename_cal, filename_test)
        elif name == "ECTSum":
            directory = DATA_PATH / 'ECTSum/'
            df_cal, df_test = load_dataset_raw_ect(directory, filename_cal, filename_test)
        elif name == "CSDS":
            directory = DATA_PATH / 'CSDS/'
            df_cal, df_test = load_dataset_raw_CSDS(directory, filename_cal, filename_test)
        else:
            raise ValueError(f"Dataset {name} download not supported, check the name to be within [tldr, MTS, ECTSum, CSDS]")

        if not use_pandas:
            df_cal = pl.from_pandas(df_cal)
            df_test = pl.from_pandas(df_test)

    if as_dataframe:
        return df_cal, df_test
    else:
        return ImportanceDataset.from_dataframe(df_cal, label_col=label_col), ImportanceDataset.from_dataframe(df_test,label_col=label_col)
    
DATASETS = ["tldr", "tldr_fulltext", "MTS", "ECTSum", "CSDS", "CNNDM"]