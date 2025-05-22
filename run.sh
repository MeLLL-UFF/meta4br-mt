#!/bin/bash
#BEFORE RUN IT, RUN ON TERMINAL ----> chmod +x run.sh

HF_TOKEN="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"
INPUT_PATH="logs/"


# MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
# OUTPUT_PATH="dataset_newsmet/llama/"
# CUDA_VISIBLE_DEVICES=0 python3 generate/genPTtoEN_locais.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "dataset_newsmet/llama/prompt2/llama-log.txt" 2>&1 

MODEL_ID="Helsinki-NLP/opus-mt-ROMANCE-en"
OUTPUT_PATH="dataset_newsmet/marian/"
CUDA_VISIBLE_DEVICES=0 python3 generate/tradPTtoEN.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "dataset_newsmet/marian/marian-log.txt" 2>&1 

MODEL_ID="facebook/nllb-200-3.3B"
OUTPUT_PATH="dataset_newsmet/meta/"
CUDA_VISIBLE_DEVICES=0 python3 generate/tradPTtoEN.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "dataset_newsmet/meta/nllb-log.txt" 2>&1 

