#!/bin/bash
#BEFORE RUN IT, RUN ON TERMINAL ----> chmod +x run.sh

HF_TOKEN="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"
INPUT_PATH="logs/"

# Traduzindo as frases pro PT

# MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
# OUTPUT_PATH="llama/"
# CUDA_VISIBLE_DEVICES=0 python3 genPTtoEN.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "llama/llama-log.txt" 2>&1 

# MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
# OUTPUT_PATH="qwen/"
# CUDA_VISIBLE_DEVICES=1 python3 genPTtoEN.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "qwen/qwen-log.txt" 2>&1 

MODEL_ID="facebook/nllb-200-3.3B"
OUTPUT_PATH="meta/"
CUDA_VISIBLE_DEVICES=1 python3 tradPTtoEN.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "meta/nllb-log.txt" 2>&1 
