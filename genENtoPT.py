from huggingface_hub import login
import argparse
import transformers
import pandas as pd
import torch
import json
import gc
import os
import re

def main(model_id, hf_token, output_path):
    login(token=hf_token)

    df = pd.read_parquet("comparacao_datasets/manual_data.parquet", engine='pyarrow') 

    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    pipeline = transformers.pipeline(
        task="text-generation",
        trust_remote_code=True,
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    os.makedirs(output_path, exist_ok=True)
    json_output_path = os.path.join(output_path, "ENtoPT.json")
    
        
    anotacoes = []
    
    for frase in df['Sentence']:

        prompt = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso"

        messages = [
            {"role": "user", "content": prompt}
        ]
        
        outputs = pipeline(
            messages,
            max_new_tokens=2000,
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

    args = parser.parse_args()

    main(args.model_id, args.hf_token, args.output_path)