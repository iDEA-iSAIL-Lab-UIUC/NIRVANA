#!/bin/bash
set -e

model=$1         
tasks=$2
nshot=$3
resultpath=$4
device=$5
batch_size=$6

shift 6 

cmd=(
  python3 main.py
  --model=hf-auto
  --model_args="pretrained=${model},dtype=bfloat16,parallelize=True"
  --tasks="${tasks}"
  --num_fewshot="${nshot}"
  --batch_size="${batch_size}"
  --output_path="../result/${resultpath}.log"
  --no_cache
  --device="${device}"
  "$@"             
)

echo "[i] Running: ${cmd[*]}"
TOKENIZERS_PARALLELISM=false "${cmd[@]}"

