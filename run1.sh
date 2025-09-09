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


# MODEL_ID="mistralai/Ministral-8B-Instruct-2410"
# OUTPUT_PATH="dataset_newsmet/mistral/"
# PROMPT_ID="prompt3"
# CUDA_VISIBLE_DEVICES=1 python3 generate/genENtoPT_locais.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH --prompt_id $PROMPT_ID > "dataset_newsmet/mistral/mistral-log.txt" 2>&1 

# MODEL_ID="mistralai/Ministral-8B-Instruct-2410"
# OUTPUT_PATH="dataset_newsmet/mistral/"
# PROMPT_ID="prompt4"
# CUDA_VISIBLE_DEVICES=1 python3 generate/genENtoPT_locais.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH --prompt_id $PROMPT_ID > "dataset_newsmet/mistral/mistral-log.txt" 2>&1 


# MODEL_ID="mistralai/Ministral-8B-Instruct-2410"
# OUTPUT_PATH="dataset_newsmet/mistral/"
# PROMPT_ID="prompt3"
# CUDA_VISIBLE_DEVICES=1 python3 generate/genPTtoEN_locais.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH --prompt_id $PROMPT_ID > "dataset_newsmet/mistral/mistral-log.txt" 2>&1 

# MODEL_ID="mistralai/Ministral-8B-Instruct-2410"
# OUTPUT_PATH="dataset_newsmet/mistral/"
# PROMPT_ID="prompt4"
# CUDA_VISIBLE_DEVICES=1 python3 generate/genPTtoEN_locais.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH --prompt_id $PROMPT_ID > "dataset_newsmet/mistral/mistral-log.txt" 2>&1

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_PATH="dataset_newsmet/qwen/"
PROMPT_ID="prompt4"
CUDA_VISIBLE_DEVICES=1 python3 generate/genENtoPT_locais.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH --prompt_id $PROMPT_ID > "dataset_newsmet/qwen/qwen-log.txt" 2>&1 

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_PATH="dataset_newsmet/qwen/"
PROMPT_ID="prompt4"
CUDA_VISIBLE_DEVICES=1 python3 generate/genPTtoEN_locais.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH --prompt_id $PROMPT_ID > "dataset_newsmet/qwen/qwen-log.txt" 2>&1 
