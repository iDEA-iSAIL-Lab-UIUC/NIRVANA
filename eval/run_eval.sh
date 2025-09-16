export CUDA_VISIBLE_DEVICES=2,6

device='cuda'

models=(   
    # '/home/mai10/NIRVANA_cleaned/model/meta-llama/Llama-3.1-8B-snip-bookcorpus-0.5'
    # '/home/mai10/NIRVANA_cleaned/model/meta-llama/Llama-3.1-8B-magnitude-bookcorpus-0.5'
    # '/home/mai10/NIRVANA_cleaned/model/meta-llama/Llama-3.1-8B-NIRVANA-bookcorpus-0.5'
    # /home/mai10/NIRVANA_cleaned/model/meta-llama/Llama-2-7b-hf-NIRVANA-bookcorpus-0.2

    # /home/mai10/NIRVANA_cleaned/model/meta-llama/Llama-3.1-8B-NIRVANA-bookcorpus-0.5
    # /home/mai10/NIRVANA_cleaned/model/meta-llama/Llama-3.1-8B-NIRVANA-bookcorpus-0.4
    # /home/mai10/NIRVANA_cleaned/model/meta-llama/Llama-3.1-8B-NIRVANA-bookcorpus-0.2

    /home/mai10/TASP_mine/lora_TASP-0.5_2
)

for model in "${models[@]}"; do
    modelname=$(basename "$model")
    echo "Running model: $modelname"
    # Zero-shot evaluation
    # bash hf_open_llm.sh $model arc_easy,winogrande 0 0shot-$modelname $device 64 --custom
    # bash hf_open_llm.sh $model hellaswag 0 add0shot-$modelname $device 32 --custom

    # MBPP 3-shot
    bash hf_open_llm.sh $model mbpp 3 mbpp3shot-$modelname $device 16 --custom
done
