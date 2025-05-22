#BEFORE RUN IT, RUN ON TERMINAL ----> chmod +x run1.sh

HF_TOKEN="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"
INPUT_PATH="logs/"


MODEL_ID="mistralai/Ministral-8B-Instruct-2410"
OUTPUT_PATH="dataset_newsmet/mistral/"
CUDA_VISIBLE_DEVICES=1 python3 generate/genPTtoEN_locais.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "dataset_newsmet/mistral/prompt2/mistral-log.txt" 2>&1 

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_PATH="dataset_newsmet/qwen/"
CUDA_VISIBLE_DEVICES=1 python3 generate/genPTtoEN_locais.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "dataset_newsmet/qwen/prompt2/qwen-log.txt" 2>&1 

