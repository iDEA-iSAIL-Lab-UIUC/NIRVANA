'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''

import os
import sys
import argparse
from typing import List
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from llama import LlamaForCausalLM, LlamaConfig
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

def main(args):
    # Set WanDB
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # Load Pruned Model    
    if 'Llama-3.1-8B' in args.base_model:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    prompter = Prompter(args.prompt_template_name)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if 'lamini' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                None,
                data_point["response"],
            )
        elif 'alpaca' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )
        else:
            raise NotImplementedError

        tokenized_full_prompt = tokenize(full_prompt)
        if not args.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"] if 'input' in data_point.keys() else None,
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=args.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if args.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def split_and_tokenizer(test_data, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(test_data[field_name]), return_tensors='pt').input_ids[0]
        nsamples = test_ids.numel() // seq_len

        test_set = []
        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_set.append({
                'input_ids': batch,
                'labels': batch
            })
        return test_set

    # Prepare For LoRA    
    model_config = LlamaConfig.from_pretrained(args.base_model, pruned=True)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        config=model_config,
        torch_dtype=torch.bfloat16,
    )
    
    pruned_dims = {name: module.weight.shape          # (out_dim, in_dim)
               for name, module in model.named_modules()
               if any(n in name for n in args.lora_target_modules.split(","))}

    # 2) Choose a rank scheme (linear‑proportional here)
    def choose_rank(out_dim, in_dim, base=32, min_r=4):
        dim = min(out_dim, in_dim)
        return max(min_r, round(base * dim / 4096))    # 512 = original width for Llama‑8B

    rank_pattern = {name: choose_rank(*dims) for name, dims in pruned_dims.items()}
    
    s = 16
    alpha_pattern  = {name: s * r for name, r in rank_pattern.items()}
    
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  
  

    # Load Train Dataset
    data = load_dataset(args.data_path)
    if args.cache_dataset and os.path.exists('datasets/cache/{}.bin'.format(args.data_path)):
        preprocess_data = torch.load('datasets/cache/{}.bin'.format(args.data_path))
        train_data, val_data = preprocess_data['train'], preprocess_data['val']
    else:
        train_val = data["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = {
            args.data_path: train_val["test"].shuffle().map(generate_and_tokenize_prompt),
        }
        if args.cache_dataset and args.local_rank == 0:
            cache_file = 'datasets/cache/{}.bin'.format(args.data_path)
            cache_dir = '/'.join(cache_file.split('/')[:-1])
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)

            torch.save({
                'train': train_data, 'val': val_data
            }, cache_file)

    # Load Extra Validation Dataset
    if args.extra_val_dataset:
        from LLMPruner.datasets.ppl_dataset import get_wikitext2, get_ptb

        seq_len = 128
        for extra_dataset in args.extra_val_dataset.split(','):
            if 'wikitext2' in extra_dataset:
                _, test_data = get_wikitext2(seq_len, None)
                test_data = split_and_tokenizer(test_data, tokenizer, seq_len, field_name='text')
            if 'ptb' in extra_dataset:
                _, test_data = get_ptb(seq_len, None)
                test_data = split_and_tokenizer(test_data, tokenizer, seq_len, field_name='sentence')
            val_data[extra_dataset] = test_data

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=50,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            bf16=True,
            fp16=False,
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            eval_strategy="steps",
            save_strategy="steps",
            torch_compile=False,             
            gradient_checkpointing=False,    
            dataloader_num_workers=0,         
            dataloader_pin_memory=False,     
            ddp_find_unused_parameters=False,
            eval_steps=args.save_steps,
            save_steps=args.save_steps,
            output_dir=args.output_dir,
            save_total_limit=20,
            load_best_model_at_end=True,
            # ddp_find_unused_parameters=None,
            group_by_length=args.group_by_length,
            report_to="none",
            run_name=args.output_dir.split('/')[-1],
            metric_for_best_model="{}_loss".format(args.data_path),
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    
    print("Start Training...")

    trainer.train()

    model.state_dict = old_state_dict
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--prune_model', type=str, help='prune model name')
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
    parser.add_argument('--cache_dataset', action="store_true", default=False)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help="The prompt template to use, will default to alpaca.")
    parser.add_argument('--no_instruction', action='store_true', default=False, help="Whether to use the instruction template or not.")
    parser.add_argument('--save_steps', type=int, default=200, help='save steps')

    # Lora Configuration
    parser.add_argument('--lora_r', type=int, default=16, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=8, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')

    # llm hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")
   
    # wandb params
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")

    #ddp
    parser.add_argument('--local_rank', type=int, default=-1)
   
    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version

    main(args)
