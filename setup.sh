#!/usr/bin/env bash
set -e

conda init bash
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create the environment
conda create -n nirvana_test python=3.9.19 -y
conda activate nirvana_test


# Install the required packages
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
cd llama
pip install -e .     
cd ../lm-evaluation-harness
pip install -e .
pip install transformers==4.49.0
pip install numpy==1.26.3