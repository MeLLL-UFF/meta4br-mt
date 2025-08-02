#!/bin/bash
#BEFORE RUN IT, RUN ON TERMINAL ----> chmod +x run.sh

CONFIG_FILE="./config.env"

# HF_TOKEN=$(head -n 1 tokens.txt) # a priemira linha é o token do HF
TOKEN="$HF_TOKEN"

if [ -z "$TOKEN" ]; then
  echo "Erro: variável de ambiente HF_TOKEN não está definida."
  exit 1
fi

INPUT_PATH="logs/"

# MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
# OUTPUT_PATH="dataset_newsmet/llama/"
# CUDA_VISIBLE_DEVICES=0 python3 generate/genPTtoEN_locais.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "dataset_newsmet/llama/prompt2/llama-log.txt" 2>&1 

# MODEL_ID="Helsinki-NLP/opus-mt-ROMANCE-en"
# OUTPUT_PATH="dataset_newsmet/marian/"
# CUDA_VISIBLE_DEVICES=0 python3 generate/tradPTtoEN.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "dataset_newsmet/marian/marian-log.txt" 2>&1 

# MODEL_ID="facebook/nllb-200-3.3B"
# OUTPUT_PATH="dataset_newsmet/meta/"
# CUDA_VISIBLE_DEVICES=0 python3 generate/tradPTtoEN.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "dataset_newsmet/meta/nllb-log.txt" 2>&1 

MODEL_ID="google/gemma-3-12b-it"
OUTPUT_PATH="dataset_newsmet/gemma3/"
python3 generate/genPTtoEN_locais.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "dataset_newsmet/gemma3/prompt2/gemma3-log.txt" 2>&1 
