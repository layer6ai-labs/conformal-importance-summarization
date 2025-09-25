import os
import requests
import re

def download_file(url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    response = requests.get(url, stream=True)
    filename = re.findall('filename="?([^"]+)"?', response.headers.get('Content-Disposition', '')) or [os.path.basename(url)]
    file_path = os.path.join(target_dir, filename[0])
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"File downloaded successfully: {file_path}")

def downloadTLDRrawdata(target_directory="./TLDR"):
    urls = [
        "https://raw.githubusercontent.com/allenai/scitldr/refs/heads/master/SciTLDR-Data/SciTLDR-AIC/train.jsonl",
        "https://raw.githubusercontent.com/allenai/scitldr/refs/heads/master/SciTLDR-Data/SciTLDR-AIC/dev.jsonl",
        "https://raw.githubusercontent.com/allenai/scitldr/refs/heads/master/SciTLDR-Data/SciTLDR-AIC/test.jsonl"
    ]
    for url in urls:
        download_file(url, target_directory)

def downloadCSDSrawdata(target_directory="./CSDS/"):
    urls = [
        "https://drive.usercontent.google.com/download?id=1JLW5iRUjdFz1BUGypyGHAm6vEkMI20sS&export=download",
        "https://drive.usercontent.google.com/download?id=1xpHJWJd5kLnq9tKqzqE-928FgYEryka1&export=download"
    ]
    for url in urls:
        download_file(url, target_directory)

def downloadECTrawdata(target_directory="./ECTSum/"):
    urls = [
        "https://huggingface.co/datasets/nyamuda/ECTSum/resolve/main/train.json?download=true",
        "https://huggingface.co/datasets/nyamuda/ECTSum/resolve/main/val.json?download=true",
        "https://huggingface.co/datasets/nyamuda/ECTSum/resolve/main/test.json?download=true"
    ]
    for url in urls:
        download_file(url, target_directory)

def downloadMTSrawdata(target_directory="./MTS/"):
    urls = [
        "https://raw.githubusercontent.com/abachaa/MTS-Dialog/refs/heads/main/Main-Dataset/MTS-Dialog-TrainingSet.csv",
        "https://raw.githubusercontent.com/abachaa/MTS-Dialog/refs/heads/main/Main-Dataset/MTS-Dialog-ValidationSet.csv",
        "https://raw.githubusercontent.com/abachaa/MTS-Dialog/refs/heads/main/Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv",
        "https://raw.githubusercontent.com/abachaa/MTS-Dialog/refs/heads/main/Main-Dataset/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv"
    ]
    for url in urls:
        download_file(url, target_directory)