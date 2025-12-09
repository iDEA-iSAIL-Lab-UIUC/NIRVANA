
import random
import numpy as np
from transformers import AutoTokenizer
from llama import LlamaForCausalLM, LlamaConfig
import argparse
import torch
from dataset.evaluator import PPLMetric
import os
from pathlib import Path
from dataset.utils import get_examples
os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

# argument for parsing
parser.add_argument('--base_model', type=str, default="meta-llama/Llama-3.1-8B", help='base model name')
parser.add_argument('--sparsity', type=float, default=0.5, help='pruning ratio')
parser.add_argument("--save_model", nargs="?", const="../model", default=None, type=Path, metavar="DIR", help="Path to save the pruned model")


# argument for generation
parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
parser.add_argument('--top_p', type=float, default=0.95, help='top p')
parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')


# Calibration data
parser.add_argument('--data', type=str, default='c4')
parser.add_argument('--data_idx', type=int, default=-1)
parser.add_argument('--num_examples', type=int, default=10)
parser.add_argument('--seq_len', type=int, default=128, help='calibration sequence length')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--select', action='store_true', help='select')
parser.add_argument('--get_kl', action='store_true', help='get kl divergence')



# argument for layer-wise pruning/column-wise pruning
parser.add_argument('--channel_wise', action='store_true', help='channel wise')
parser.add_argument('--block_wise', action='store_true', help='block wise')
parser.add_argument('--layer_wise', action='store_true', help='layer wise')
parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=0)
parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

# Pruner
parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
parser.add_argument('--prune_type', type=str, default='NIRVANA', help='choose from [vectorize, param_second, param_first, param_mix]')


# general argument
parser.add_argument('--device', type=str, default="cuda", help='device')
parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
parser.add_argument('--test_after_train', action='store_true', help='whether test after train')
parser.add_argument('--test_after_prune', action='store_true', help='whether test after prune')
parser.add_argument('--prune', action='store_true', help='whether prune')
parser.add_argument('--gamma', type=float, default=1.0, help='Attn vs MLP scaling factor')
parser.add_argument('--seed', type=int, default=42, help='seed')

args = parser.parse_args()

torch_version = float('.'.join(torch.__version__.split('.')[:2]))
args.torch_version = torch_version



def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def structure_prune_weight(linear_layer, neuron_indices, feature='row'):
      # Calculate the L1-norm of each neuron's weights        
      weight_mask = torch.ones_like(linear_layer.data, dtype=torch.bool)

      if feature == 'row':
        weight_mask[neuron_indices] = False
        
        linear_layer = linear_layer[weight_mask].clone().detach().view(
            linear_layer.size(0) - neuron_indices.numel(), linear_layer.size(1)
        )
      elif feature == 'column':
        weight_mask[:, neuron_indices] = False
        linear_layer = linear_layer[weight_mask].clone().detach().view(
          linear_layer.size(0), -1
        )
        
      
      del neuron_indices, weight_mask
      torch.cuda.empty_cache()
        
      
      return linear_layer
  
  
def get_pruning_idx(score, prune_type, n_pruned=-1, threshold=-1, global_pruning=False, head_dim:int=0, attn_type=None, kv_groups:int=0):

  if global_pruning:
    if threshold < 0:
      raise ValueError("Threshold must be set for global pruning")
    pruning_indices = torch.where(score<=threshold)[0]
  else:
    if n_pruned < 0:
      raise ValueError("Number of pruned neurons must be set for local pruning")
    _, pruning_indices = torch.topk(score, k=n_pruned, largest=False)
    
  if len(pruning_indices) == 0:return pruning_indices
  
  if len(pruning_indices) == len(score):pruning_indices=pruning_indices[:-1]
  
  
  if prune_type == 'attn':
    pruning_indices = torch.cat(
              [torch.tensor([j+head_dim*i for j in range(head_dim)])
              for i in pruning_indices], 0)
    
    if attn_type == 'q_proj' or attn_type == 'o_proj':
      pruning_indices = torch.cat(
              [torch.tensor([j+kv_groups*i for j in range(kv_groups)])
              for i in pruning_indices], 0)
      
  else:
    candidate_indices = torch.where(score <= threshold)[0]
    
    n_total = len(score)
    n_kept_initial = n_total - len(candidate_indices)

    # Strategy: Ensure the number of KEPT units is a multiple of 8.
    n_kept_final = ((n_kept_initial + 7) // 8) * 8
    n_kept_final = min(n_total, n_kept_final) # Cannot keep more than total

    # If everything is getting pruned, ensure we keep at least one block of 8.
    if n_kept_final == 0 and n_total > 0:
          n_kept_final = 8

    n_to_spare = n_kept_final - n_kept_initial

    if n_to_spare > 0:
        # We need to spare some units.
        if n_to_spare >= len(candidate_indices):
            pruning_indices = torch.tensor([], dtype=torch.long, device=score.device)
        else:
            candidate_scores = score[candidate_indices]
            
            # Find the top scores among the candidates to be spared.
            _, relative_indices_to_spare = torch.topk(candidate_scores, k=n_to_spare, largest=True)
            
            # Get their original indices in the full score tensor.
            original_indices_to_spare = candidate_indices[relative_indices_to_spare]
            
            pruning_mask = torch.zeros_like(score, dtype=torch.bool)
            pruning_mask[candidate_indices] = True
            pruning_mask[original_indices_to_spare] = False
            
            pruning_indices = torch.where(pruning_mask)[0]
    else:
        pruning_indices = candidate_indices
  return pruning_indices
  



def main(args):
  

  tokenizer = AutoTokenizer.from_pretrained(args.base_model)
  
  example_prompts = get_examples(args.data,
                                 tokenizer, 
                                 args.num_examples, 
                                 seq_len=args.seq_len,
                                 idx=args.data_idx,
                                 verbose=args.verbose,
                                 select=args.select,
                                 seed=args.seed,
                                  ).to(args.device)
  
  if example_prompts is None:
    print("No examples found")
    return
  
  

  model_config = LlamaConfig.from_pretrained(args.base_model)

  
  model = LlamaForCausalLM.from_pretrained(
    args.base_model,
    config=model_config,
    low_cpu_mem_usage=True if args.torch_version >=1.9 else False,
    device_map='auto',
    torch_dtype=torch.bfloat16
  )
  
  before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
  
  
  
  
  if args.prune:
    
    print("Start pruning with {}".format(args.prune_type))    
    print("Start Backwarding in iterative steps = {}...".format(1))
  
  
  
    if args.prune_type == 'NIRVANA':
      output_orig = model(example_prompts, labels=example_prompts,return_dict=True)['logits']
      output_logits = torch.mean(torch.max(output_orig, dim=1)[0])
      output_logits.backward()
    
    
    num_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    kv_groups = model.config.num_attention_heads // model.config.num_key_value_heads
    
    score_norms = {}
    
          
    
    for m, p in list(model.named_parameters()):   
      if 'self_attn' in m:
        
        
        layer_name, param_name = m.rsplit('.', 1)
        layer = dict(model.named_modules())[layer_name]
        
          
        if '.weight' in m:
            param_name = '.'.join(m.split('.')[:-2])
            if args.prune_type == 'NIRVANA':
              score = torch.clone(p.grad * p.data).detach().abs_()
            elif args.prune_type == 'magnitude':
              score = torch.clone(p.data).detach().abs_()
            if m.split('.')[-2] in ['q_proj', 'k_proj', 'v_proj']:
              score = torch.norm(score, p=1, dim=1)
              
              # Aggregate the scores for each head.
              # Here we sum over both the head dimension and the remaining dimensions.
              score = score.view(num_heads, head_dim, -1)
              head_score = score.sum(dim=(1, 2))
              score_norms[param_name] = score_norms.get(param_name,0) + head_score
              
        
            elif m.split('.')[-2] in ['o_proj']:
              
              score = torch.norm(score, p=1, dim=0)
              
              # Aggregate the scores for each head.
              # Here we sum over both the head dimension and the remaining dimensions.
              score = score.view(num_heads, head_dim, -1)
              head_score = score.sum(dim=(1, 2))
              score_norms[param_name] = score_norms.get(param_name,0) + head_score
              
      elif 'mlp' in m:
        
        if args.prune_type == 'NIRVANA':
              score = torch.clone(p.grad * p.data).detach().abs_()
        elif args.prune_type == 'magnitude':
          score = torch.clone(p.data).detach().abs_()
        
        if '.weight' in m:
            if m.split('.')[-3] == 'mlp':
                param_name = '.'.join(m.split('.')[:-2])
                if m.split('.')[-2] in ['gate_proj', 'up_proj']:
                  score_norms[param_name] = score_norms.get(param_name,0) + torch.norm(score, p=1, dim=1)
            
                elif m.split('.')[-2] in ['down_proj']:
                  score_norms[param_name] = score_norms.get(param_name,0) + torch.norm(score, p=1, dim=0)        
              
              
    
    attn_score = torch.cat([torch.flatten(v).to('cpu') for n,v in score_norms.items() if 'self_attn' in n])   
    mlp_score = torch.cat([torch.flatten(v).to('cpu') for n,v in score_norms.items() if 'mlp' in n])
    
    
    gamma = 1.0/args.gamma
    
    orginal_pruned = (head_dim * 2 * (kv_groups + 1) * num_heads + model.config.intermediate_size * 3) * args.sparsity  
    attn_mlp_ratio = (head_dim * 2 * (kv_groups + 1) * num_heads)/(model.config.intermediate_size * 3) * gamma * 3 / (head_dim * 2 * (kv_groups+1))  
    now_pruend = (head_dim * 2 * (kv_groups +1 ) + 1/attn_mlp_ratio * 3)   
    prune_attn = orginal_pruned/now_pruend*len(model.model.layers)
    prune_mlp = int(prune_attn/attn_mlp_ratio)
    prune_attn = int(prune_attn)
    
    
    if prune_attn>0:
      topk_imp_attn, _ = torch.topk(attn_score, k=prune_attn, largest=False)
    topk_imp_mlp, _ = torch.topk(mlp_score, k=prune_mlp, largest=False)
    
    attn_threshold, mlp_threshold = topk_imp_attn[-1] if prune_attn>0 else 0, topk_imp_mlp[-1]
    
    
    for m, p in list(model.named_parameters()):
      layer_name, param_name = m.rsplit('.', 1)
      layer = dict(model.named_modules())[layer_name]
      
      if 'self_attn' in m:
        
        attn_type = m.split('.')[-2]
        
        head_score = score_norms['.'.join(m.split('.')[:-2])]
        if attn_type in ['q_proj', 'k_proj', 'v_proj']:              
          
          
          prune_idx = get_pruning_idx(score=head_score,
                                      threshold=attn_threshold, 
                                      global_pruning=True,
                                      prune_type='attn',
                                      attn_type=attn_type,
                                      head_dim=head_dim,
                                      kv_groups=kv_groups)
          
          if len(prune_idx) == 0:continue
          
          pruned_param = structure_prune_weight(p, 
                                                prune_idx,
                                                'row')
          
          
          
          delattr(layer, param_name)
          layer.out_features = pruned_param.size(0)
          
          layer.register_parameter(param_name, torch.nn.Parameter(pruned_param)) 
          
        elif attn_type in ['o_proj']:
          
          prune_idx = get_pruning_idx(score=head_score,
                                      threshold=attn_threshold, 
                                      global_pruning=True,
                                      prune_type='attn',
                                      attn_type=attn_type,
                                      head_dim=head_dim,
                                      kv_groups=kv_groups)
          
          if len(prune_idx) == 0:continue
          
          pruned_param = structure_prune_weight(p, 
                                                prune_idx,
                                                'column')
          
          delattr(layer, param_name)
          layer.in_features = pruned_param.size(1)
          
          layer.register_parameter(param_name, torch.nn.Parameter(pruned_param)) 
        
      elif 'mlp' in m:
        if '.weight' in m:
          if m.split('.')[-3] == 'mlp':
                prune_idx = get_pruning_idx(score=score_norms['.'.join(m.split('.')[:-2])],
                                              threshold=mlp_threshold, 
                                              global_pruning=True,
                                              prune_type='mlp'
                                              )
                
                if m.split('.')[-2] in ['gate_proj', 'up_proj']:
                  
                  pruned_param = structure_prune_weight(p, 
                                                        feature='row',
                                                        neuron_indices=prune_idx)
                  
                  delattr(layer, param_name)
                  layer.out_features = pruned_param.size(0)
            
                elif m.split('.')[-2] in ['down_proj']:
                  
                  pruned_param = structure_prune_weight(p, 
                                                        feature='column',
                                                        neuron_indices=prune_idx)
                  
                  delattr(layer, param_name)
                  layer.in_features = pruned_param.size(1)
                  
                layer.register_parameter(param_name, torch.nn.Parameter(pruned_param))
                      



        
  
  for idx, layer in enumerate(model.model.layers):    
      layer.self_attn.num_attention_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
      

        
  after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
  
  if args.get_kl:          
      ori_model_config = LlamaConfig.from_pretrained(args.base_model)
      ori_model = LlamaForCausalLM.from_pretrained(
          args.base_model,
          config=ori_model_config,
          low_cpu_mem_usage=True,
          device_map='auto',
          torch_dtype=torch.bfloat16
      )

      test_data = get_examples(args.data,
                                tokenizer, 
                                n_samples=128, 
                                seq_len=64,
                                seed=42
                                ).to('cuda')
      

      kl_loss = compute_influence_loss(original_model=ori_model, pruned_model=model, test_data=test_data)
      print('KL divergence:', kl_loss)
      
      del ori_model


  
  
  if args.test_after_prune:
    ppl = PPLMetric(model, tokenizer, ['wikitext', 'ptb','lambada'], args.max_seq_len, device=args.eval_device)
    print("PPL after pruning: {}".format(ppl))
      
  if args.save_model:
    for idx, layer in enumerate(model.model.layers): 
      model.config.modified_intermediate_dimension.append(layer.mlp.gate_proj.weight.shape[0])
      model.config.modified_head_num.append(layer.self_attn.num_attention_heads)
      
    model.config.pruned=True
    model.config.pruned_attn=True
    
    save_dir = args.save_model / f"{args.base_model}-" \
               f"{args.prune_type.split('_')[-1]}-{args.data}-{args.sparsity}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)






 
set_all_seeds(42)    
main(args)