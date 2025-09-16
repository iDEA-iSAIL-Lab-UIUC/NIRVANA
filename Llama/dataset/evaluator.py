'''
Some of the code refer to
https://github.com/IST-DASLab/gptq/blob/main/datautils.py
'''

import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data.dataset import Dataset

def get_wikitext(seq_len, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-103-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-103-v1', split='test')
    return traindata, testdata

def get_ptb(seq_len, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    return traindata, valdata
  
def get_lambada(seq_len, tokenizer):
    testdata = load_dataset("EleutherAI/lambada_openai", "en", split='test')
    return testdata
  
def get_bookcorpus(seq_len, tokenizer, num_samples=10000):
    traindata = load_dataset(
        'bookcorpus', split='train', trust_remote_code=True
    )
    
    traindata = traindata.filter(lambda x: len(x["text"]) > seq_len)
    
    return traindata.select(range(num_samples))

class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

def process_data(samples, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)
       

def get_loaders(name, tokenizer, seq_len=2048, batch_size = 8):
    if 'wikitext' in name:
        train_data, test_data = get_wikitext(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        train_data, test_data = get_ptb(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    if 'lambada' in name:
        train_data = ''
        test_data = get_lambada(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'bookcorpus' in name:
        train_data = ''
        test_data = get_bookcorpus(seq_len, tokenizer, num_samples=512)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
        
    # print(test_data)      

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_data, test_loader

def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size = 4, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        ppl = llama_eval(model, test_loader, device)
        metric[dataset] = ppl
        # print(metric)
    return metric
  
  
@torch.no_grad()
def compute_influence_loss(original_model, pruned_model, test_data, device="cuda", eps: float = 1e-12):
    # _, test_data = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
    # original_model.eval()
    # pruned_model.eval()
    test_data = [test_data]
    
    total_kl = 0.0
    n_batches = 0
    for z in tqdm(test_data):
        # forward
        
        # with torch.no_grad():
        z = z.to(device)
        
        logits_p = original_model(z).logits.float().to('cpu')
        logits_q = pruned_model(z).logits.float().to('cpu')

        # stable log‑probs
        log_p = logits_p.log_softmax(dim=-1)
        log_q = logits_q.log_softmax(dim=-1)

        # probs in FP32 (will be >= eps)
        p = log_p.exp().clamp(min=eps)

        # D_KL(P || Q) = sum_x P(x) [log P(x) - log Q(x)]
        # sum over vocab, then mean over tokens & batch
        kl_per_token = (p * (log_p - log_q)).sum(dim=-1)  
        kl = kl_per_token.mean()  

        # print(kl)

        # # accumulate as float
        total_kl += kl.item()
        n_batches += 1

        # clean up (optional—should be freed by the next loop iteration)
        del logits_p, logits_q, log_p, log_q, kl
        torch.cuda.empty_cache()
      
            
    return total_kl / n_batches if n_batches else 0.0

@torch.no_grad()
def llama_eval(model, test_lodaer, device):
    nlls = []
    n_samples = 0
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits
    
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    #print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()