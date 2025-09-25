import pandas as pd
import json
from nltk import sent_tokenize
from tqdm import tqdm

tqdm.pandas()

def combine_conversations(conversation):
    sentences = [s.strip() for s in conversation.replace('\r', '').split('\n') if s.strip()]
    combined, current = [], ""
    for sentence in sentences:
        if sentence.startswith("Doctor:"):
            if current: combined.append(current)
            current = sentence
        elif sentence.startswith("Patient:"):
            current += f" {sentence}" if current else sentence
        else:
            current += f" {sentence}"
    if current: combined.append(current)
    return [s.strip() for s in combined]

def process_QA(sample):
    indices = {idx for item in sample['QA'] for idx in item['QueSummUttIDs'] + item['AnsSummLongUttIDs']}
    return [1 if i in indices else 0 for i in range(len(sample['Dialogue']))]

def process_dialogue(sample):
    return [f"{'客服' if item['speaker'] == 'A' else '用户'} : {item['utterance']}" for item in sample.get('Dialogue', '')]

def process_final_summary(sample):
    return sample['FinalSumm']

def composeCSDSdf(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return pd.DataFrame({
        'input': [' '.join(process_dialogue(s)) for s in data],
        'summary': [' '.join(process_final_summary(s)) for s in data],
        'input_sentences': [process_dialogue(s) for s in data],
        'summary_sentences': [process_final_summary(s) for s in data],
        'input_sentences_labels': [process_QA(s) for s in data],
    })

def processTLDRRawData(df):
    df = df.rename(columns={'source': 'input_sentences', 'target': 'summary_sentences'})
    df = df[df['summary_sentences'].str.len() > 1]
    df['input'] = df['input_sentences'].str.join(' ')
    df['summary'] = df['summary_sentences'].str.join(' ')
    return df[['input', 'summary', 'input_sentences', 'summary_sentences']]

def processECTRawData(df):
    df = df.rename(columns={'summaries': 'summary', 'doc': 'input', 'labels': 'input_sentences_labels'})
    df['input_sentences'] = df['input'].str.split('\n')
    df['summary_sentences'] = df['summary'].str.split('\n')
    df['input_sentences_labels'] = df['input_sentences_labels'].str.split('\n').apply(lambda x: list(map(int, x)))
    return df

def processMTSRawData(df):
    df = df.rename(columns={'section_text': 'summary', 'dialogue': 'input'})
    df['input_sentences'] = df['input'].apply(combine_conversations)
    df = df[df['input_sentences'].str.len() > 2]
    df['summary_sentences'] = df['summary'].apply(sent_tokenize)
    return df[['input', 'summary', 'input_sentences', 'summary_sentences']]