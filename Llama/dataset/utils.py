from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch

import random
import numpy as np







def get_c4(tokenizer, n_samples, seq_len, select=False, idx=-1, verbose=False):
    # traindata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    # )
    
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    
    tokenized_samples, history = [], []
    
    
    if idx>=0:      
        tokenized_sample = tokenizer(traindata[idx]['text'], return_tensors='pt')
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
        
        # print(tokenizer.decode(tokenized_sample.input_ids[:, i:i+seq_len][0]))
    
    else:
      if select:
        for idx, i in enumerate(i_list):
            if idx >= n_samples:break
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
            # print(i)
            tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
      else:
        for _ in range(n_samples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
                if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                    history.append(i)
                    break
            # print(i)
            i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
            # print(traindata[i])
            if verbose:
              print(tokenizer.decode(tokenized_sample.input_ids[:, i:i+seq_len][0]))
            tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
          
    # print(tokenized_samples)
          
    return torch.cat(tokenized_samples, dim=0)
  
def get_lamb(tokenizer, n_samples, seq_len, select=False, idx=-1):
    
    traindata = load_dataset("EleutherAI/lambada_openai", "en", split='test')
    
    
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        # print(traindata[i])
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)
  
def get_mbpp(tokenizer, n_samples, seq_len, select=False, idx=-1):
    
    traindata = load_dataset("google-research-datasets/mbpp", "full", split='test')
    
    
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text']+traindata[i]['code'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        # print(traindata[i])
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_gsm8k(tokenizer, n_samples, seq_len, select=False, idx=-1):
    
    traindata = load_dataset("openai/gsm8k", "main", split='train')
    
    
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['question']+'\n'+traindata[i]['answer'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        # print(traindata[i])
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_bookcorpus(tokenizer, n_samples, seq_len, select=False, idx=-1, verbose=False, seed=42):
    traindata = load_dataset(
        'bookcorpus', split='train', trust_remote_code=True
    )
    
    random.seed(seed)
    
    # print(seq_len)
    
    if idx>=0: 
      traindata = traindata.filter(lambda x: len(x["text"]) > 1024)
    else:
      traindata = traindata.filter(lambda x: len(x["text"]) > seq_len)
    
    
    tokenized_samples, history = [], []
    
    if idx>=0:       
        # print('idx', idx)
        tokenized_sample = tokenizer(traindata[idx]['text'], return_tensors='pt')
        if tokenized_sample.input_ids.shape[1] - seq_len < 0:
          print('too short')
          print(traindata[idx]['text'])
          print(tokenized_sample.input_ids)
          return None
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])        
        if verbose:  
          print(tokenizer.decode(tokenized_sample.input_ids[:, i:i+seq_len][0]))
    
    else:        
        for _ in range(n_samples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
                if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                    history.append(i)
                    break
            if verbose:
              print(tokenizer.decode(tokenized_sample.input_ids[:, i:i+seq_len][0]))
            i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)        
            tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
          # print(tokenizer.decode(tokenized_sample.input_ids[:, i:i+seq_len][0]))
    return torch.cat(tokenized_samples, dim=0 )

def get_wikitext(tokenizer, n_samples, seq_len, select=False, idx=-1, verbose=False):
    traindata = load_dataset(
        'wikitext','wikitext-103-v1',
        split='train'
    )
    
    filtered_dataset = traindata.filter(lambda x: len(x["text"]) > seq_len)
    
    tokenized_samples, history = [], []
    
    if idx>=0:  
        # print('idx', idx)
        tokenized_sample = tokenizer(filtered_dataset[idx]['text'], return_tensors='pt')
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])        
        if verbose:
          print(tokenizer.decode(tokenized_sample.input_ids[:, i:i+seq_len][0]))
    
    else:
        for _ in range(n_samples):
            # while True:
            for i in range(0, len(filtered_dataset) - 1):
            # for i in choose_set:
                # i = random.randint(0, len(traindata) - 1)
                tokenized_sample = tokenizer(filtered_dataset[i]['text'], return_tensors='pt')
                # print(tokenized_sample.input_ids.shape)
                if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                    # print(i)
                    history.append(i)
                    # print(filtered_dataset[i])
                    break
            # print(i)
            i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
            
            tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
          
      # print(history)
    return torch.cat(tokenized_samples, dim=0)
  

def get_examples(dataset, tokenizer, n_samples, seq_len = 128, select=False, idx=-1, verbose=False, seed=42):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len, select, idx, verbose)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len, select, idx, verbose, seed)
    elif dataset == 'wikitext':
        return get_wikitext(tokenizer, n_samples, seq_len, select, idx, verbose, seed)
    elif dataset == 'lambada':
        return get_lamb(tokenizer, n_samples, seq_len, select, idx, verbose)
    elif dataset == 'gsm8k':
        return get_gsm8k(tokenizer, n_samples, seq_len, select, idx, verbose)
    elif dataset == 'mbpp':
        return get_mbpp(tokenizer, n_samples, seq_len, select, idx, verbose)
    else:
        raise NotImplementedError


