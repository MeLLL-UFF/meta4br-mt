from huggingface_hub import login
import argparse
import transformers
import pandas as pd
import torch
import json
import gc
import os
import re

def main(model_id, hf_token, output_path, prompt_id):
    login(token=hf_token)

    dataset_name = output_path.split("_", 1)[1].split("/")[0]

    if dataset_name == "newsmet":
        df = pd.read_csv("comparacao_datasets/newsmet.csv")
        term = "Text"
    elif dataset_name == "manual_data":
        df = pd.read_parquet("comparacao_datasets/manual_data.parquet")
        term = "Sentence"

    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    pipeline = transformers.pipeline(
        task="text-generation",
        trust_remote_code=True,
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
    )
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    os.makedirs(output_path, exist_ok=True)
    json_output_path = os.path.join(output_path, f"{prompt_id}/ENtoPT.json")
    
        
    anotacoes = []
    
    for frase in df[term]:

        match prompt_id:
            case "prompt1":
                prompt = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso"
            case "prompt2":
                prompt = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso. A frase pode ou não conter metáfora"
            case "prompt3":
                prompt = f"Você é um especialista em metáforas e tradução criativa. Traduza {frase} para o português, mantendo o sentido metafórico original. Responda apenas com a tradução."
            case "prompt4":
                prompt = f"Você é um especialista em metáforas e tradução criativa. Somente traduza {frase} para o português, mantendo o sentido metafórico original. Por exemplo, 'kick the bucket' deve ser traduzido como 'bater as botas', e não como 'chutar o balde'. Responda apenas com a tradução."

        messages = [
            {"role": "user", "content": prompt}
        ]
        
        outputs = pipeline(
            messages,
            max_new_tokens=200,
            do_sample=True,
            temperature=1,
            top_p=0.95,
        )

        generated_texts = outputs[0]["generated_text"][1]['content']
  
        result = {
            "fraseEN" : frase,        
            "traducaoPT": generated_texts
        }
        
        anotacoes.append(result)
          
        torch.cuda.empty_cache()
        gc.collect()
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(anotacoes, f, ensure_ascii=False, indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text generation with Hugging Face pipeline.")
    parser.add_argument('--model_id', type=str, required=True, help="Hugging Face Model ID.")
    parser.add_argument('--hf_token', type=str, required=True, help="Hugging Face API token.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the generated outputs.")
    parser.add_argument('--prompt_id', type=str, required=True, help="Número do prompt a ser usado.")


    args = parser.parse_args()

    main(args.model_id, args.hf_token, args.output_path, args.prompt_id)