import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os
import random
import numpy as np




def load_language_model(args, baseline:int=0):
    """
    baseline:
        BERT
    best:
        sentence-transformers/all-MiniLM-L6-v2
    """
    if baseline:
        model = AutoModel.from_pretrained(f"bert-base-uncased")
    else:
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
    
    return model.to(args.device)

def load_tokenizer(args, baseline:int=0):
    """
    baseline:
        BERT
    best:
        sentence-transformers/all-MiniLM-L6-v2
    """
    if baseline:
        tokenizer = AutoTokenizer.from_pretrained(f"bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2') # 제일 잘 됨
    
    return tokenizer

def text_emebdding(args, input_text:str, tokenizer, model):
    """
    input_text: string of the text
    """
    encoded_input = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt').to(args.device)

    with torch.no_grad():
        model_output = model(encoded_input)
        embedding = model_output
    
    return torch.stack(embedding, dim=0)

def setSeeds(seed=42):

    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def tokenize(args, text:str, tokenizer, max_length:int=120):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    encoded_input = tokenizer.encode(
                text,
                add_special_tokens=True,
                return_tensors='pt', # 설정하면 (120) shape의 텐서로 저장함
                padding="max_length",
                max_length=max_length,
                truncation=True,
                )
    return encoded_input

class TQDMBytesReader(object):

    def __init__(self, fd, **kwargs):
        self.fd = fd
        from tqdm import tqdm
        self.tqdm = tqdm(**kwargs)

    def read(self, size=-1):
        bytes = self.fd.read(size)
        self.tqdm.update(len(bytes))
        return bytes

    def readline(self):
        bytes = self.fd.readline()
        self.tqdm.update(len(bytes))
        return bytes

    def __enter__(self):
        self.tqdm.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        return self.tqdm.__exit__(*args, **kwargs)