export CUDA_VISIBLE_DEVICES=2


type=NIRVANA

data=bookcorpus

num_examples=32
batch_size=32
seq_len=128
sparsity=0.5
# sparsity=0.4
# sparsity=0.2


args=(
--base_model meta-llama/Llama-3.1-8B
# --base_model meta-llama/Llama-3.2-3B

--device cuda


# Prune setting
--prune_type $type
--prune

# Calibration setting
--data $data
--num_examples $num_examples
--seq_len $seq_len
--batch_size $batch_size
--max_seq_len 512
--sparsity $sparsity
--seed 12 # Data seed for Llama-3.1-8B
# --seed 5 # Data seed for Llama-3.2-3B


# Model setting
--gamma 3.36 # Default gamma value for Llama-3.1-8B
# --gamma 3.0 # Default gamma value for Llama-3.2-3B
--test_after_prune
--save_model ../model
)


python main.py "${args[@]}"