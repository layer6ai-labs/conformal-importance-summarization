import pandas as pd
from src.data.processing_utils import composeCSDSdf

def loadECTrawdata(path):
    df_train = pd.read_json(f'{path}/train.json', lines=True)
    df_val = pd.read_json(f'{path}/val.json', lines=True)
    df_test = pd.read_json(f'{path}/test.json', lines=True)
    return pd.concat([df_train, df_val, df_test])

def loadCSDSrawdata(path):
    df_val = composeCSDSdf(f"{path}/val.json")
    df_test = composeCSDSdf(f"{path}/test.json")
    return pd.concat([df_val, df_test])

def loadMTSrawdata(path):
    df_train = pd.read_csv(f'{path}/MTS-Dialog-TrainingSet.csv')
    df_val = pd.read_csv(f'{path}/MTS-Dialog-ValidationSet.csv')
    df_test1 = pd.read_csv(f'{path}/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv')
    df_test2 = pd.read_csv(f'{path}/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv')
    return pd.concat([df_train, df_val, df_test1, df_test2])

def loadTLDRrawdata(path):
    df_train = pd.read_json(f'{path}/train.jsonl', lines=True)
    df_val = pd.read_json(f'{path}/dev.jsonl', lines=True)
    df_test = pd.read_json(f'{path}/test.jsonl', lines=True)
    return pd.concat([df_train, df_val, df_test])
