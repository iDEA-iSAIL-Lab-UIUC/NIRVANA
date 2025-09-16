export CUDA_VISIBLE_DEVICES=2

model=../model/meta-llama/Llama-3.1-8B-NIRVANA-bookcorpus-0.5 # Put your path here

OUTDIR=../model/NIRVANA-0.5-lora-alpaca # Finetuned model output directory
export NCCL_P2P_DISABLE=1        # good practice on multi‑GPU single node
export TOKENIZERS_PARALLELISM=false
# export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1


python post_training.py \
  --base_model $model \
  --output_dir $OUTDIR            \
  --batch_size 16                 \
  --micro_batch_size 2            \
  --learning_rate 1e-4            \
  --num_epochs 2                  \
  --cutoff_len 2048               \
  --save_steps 1000               \
  --lora_r 8 --lora_alpha 16      \
  # $RESUME_ARG

