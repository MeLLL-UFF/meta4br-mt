from huggingface_hub import login
import argparse
import pandas as pd
import torch
import json
import gc
import os
import transformers

# pip install openai
# pip install pandas
# pip install torch
# pip install huggingface_hub

def main(model_id, hf_token, output_path, prompt_id):
    login(token=hf_token)   

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
    arquivo_entrada = os.path.join(output_path, f"{prompt_id}/ENtoPT.json")
    arquivo_saida = os.path.join(output_path, f"{prompt_id}/PTtoEN.json")

    anotacoes = []

    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        vetor = json.load(f)

    for objeto in vetor:

        match prompt_id:
            case "prompt1":
                prompt = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso"
            case "prompt2":
                prompt = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso. A frase pode ou não conter metáfora"
            case "prompt3":
                prompt = f"Você é um especialista em metáforas e tradução criativa. Traduza '{objeto['traducaoPT']}' para o inglês, mantendo o sentido metafórico original. Responda apenas com a tradução."
            case "prompt4":
                prompt = f"Você é um especialista em metáforas e tradução criativa. Somente traduza '{objeto['traducaoPT']}' para o inglês, mantendo o sentido metafórico original. Por exemplo, 'kick the bucket' deve ser traduzido como 'bater as botas', e não como 'chutar o balde'. Responda apenas com a tradução."

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
            "frasePT": objeto['traducaoPT'],
            "traducaoEN": generated_texts
        }
            
        anotacoes.append(result)
            
        torch.cuda.empty_cache()
        gc.collect()

        # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
        with open(arquivo_saida, 'w', encoding='utf-8') as f:
            json.dump(anotacoes, f, ensure_ascii=False, indent=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text generation with Hugging Face pipeline.")
    parser.add_argument('--model_id', type=str, required=True, help="Hugging Face Model ID.")
    parser.add_argument('--hf_token', type=str, required=True, help="Hugging Face API token.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the generated outputs.")
    parser.add_argument('--prompt_id', type=str, required=True, help="Número do prompt.")

    args = parser.parse_args()

    main(args.model_id, args.hf_token, args.output_path, args.prompt_id)



