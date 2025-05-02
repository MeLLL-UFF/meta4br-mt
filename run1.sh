#BEFORE RUN IT, RUN ON TERMINAL ----> chmod +x run1.sh

HF_TOKEN="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"
INPUT_PATH="logs/"


MODEL_ID="mistralai/Ministral-8B-Instruct-2410"
OUTPUT_PATH="dataset_manualdata/mistral/"
CUDA_VISIBLE_DEVICES=1 python3 generate/genPTtoEN_locais.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "dataset_manualdata/mistral/mistral-log.txt" 2>&1 

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_PATH="dataset_manualdata/qwen/"
CUDA_VISIBLE_DEVICES=1 python3 generate/genPTtoEN_locais.py --model_id $MODEL_ID --hf_token $HF_TOKEN --output_path $OUTPUT_PATH > "dataset_manualdata/qwen/qwen-log.txt" 2>&1 
