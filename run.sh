#!/bin/bash
#BEFORE RUN IT, RUN ON TERMINAL ----> chmod +x run.sh

HF_TOKEN="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"
INPUT_PATH="logs/"

# Traduzindo as frases pro PT

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_PATH="BackTranslation/llama/"
CUDA_VISIBLE_DEVICES=0 python3 BackTranslation/genENtoPT.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "BackTranslation/llama/llama-log.txt" 2>&1 

# MODEL_ID="mistralai/Ministral-8B-Instruct-2410"
# OUTPUT_PATH="BackTranslation/mistral/"
# CUDA_VISIBLE_DEVICES=1 python3 BackTranslation/genENtoPT.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "BackTranslation/mistral/mistral-log.txt" 2>&1 

# MODEL_ID="google/gemma-2-9b-it"
# OUTPUT_PATH="BackTranslation/gemma/"
# CUDA_VISIBLE_DEVICES=0 python3 BackTranslation/genENtoPT.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "BackTranslation/gemma/gemma-log.txt" 2>&1 

# MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
# OUTPUT_PATH="BackTranslation/qwen/"
# CUDA_VISIBLE_DEVICES=1 python3 BackTranslation/genENtoPT.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "BackTranslation/qwen/qwen-log.txt" 2>&1 

# Traduzindo as frases pro EN

# MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
# OUTPUT_PATH="BackTranslation/llama/"
# CUDA_VISIBLE_DEVICES=0 python3 BackTranslation/genPTtoEN.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "BackTranslation/llama/llama-log.txt" 2>&1 

# MODEL_ID="mistralai/Ministral-8B-Instruct-2410"
# OUTPUT_PATH="BackTranslation/mistral/"
# CUDA_VISIBLE_DEVICES=1 python3 BackTranslation/genPTtoEN.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "BackTranslation/mistral/mistral-log.txt" 2>&1 

# MODEL_ID="google/gemma-2-9b-it"
# OUTPUT_PATH="BackTranslation/gemma/"
# CUDA_VISIBLE_DEVICES=0 python3 BackTranslation/genPTtoEN.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "BackTranslation/gemma/gemma-log.txt" 2>&1 

# MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
# OUTPUT_PATH="BackTranslation/qwen/"
# CUDA_VISIBLE_DEVICES=1 python3 BackTranslation/genPTtoEN.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "BackTranslation/qwen/qwen-log.txt" 2>&1 