#BEFORE RUN IT, RUN ON TERMINAL ----> chmod +x run1.sh

HF_TOKEN="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"
INPUT_PATH="logs/"

# Traduzindo as frases pro PT

# MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
# OUTPUT_PATH="llama/"
# CUDA_VISIBLE_DEVICES=0 python3 genENtoPT.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "BackTranslation/llama/llama-log.txt" 2>&1 

# MODEL_ID="mistralai/Ministral-8B-Instruct-2410"
# OUTPUT_PATH="mistral/"
# CUDA_VISIBLE_DEVICES=1 python3 genPTtoEN.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "mistral/mistral-log.txt" 2>&1 

MODEL_ID="Helsinki-NLP/opus-mt-en-ROMANCE"
OUTPUT_PATH="marian/"
CUDA_VISIBLE_DEVICES=0 python3 tradENtoPT.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "marian/marian-log.txt" 2>&1 

MODEL_ID="Helsinki-NLP/opus-mt-ROMANCE-en"
OUTPUT_PATH="marian/"
CUDA_VISIBLE_DEVICES=0 python3 tradPTtoEN.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "marian/marian-log.txt" 2>&1 
