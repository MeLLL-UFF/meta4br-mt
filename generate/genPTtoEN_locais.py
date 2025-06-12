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

def main(model_id, hf_token, output_path):
    login(token=hf_token)   

    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    pipeline = transformers.pipeline(
        task="text-generation",
        trust_remote_code=True,
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        # device=device,
        device_map="auto",
    )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    os.makedirs(output_path, exist_ok=True)
    arquivo_entrada = os.path.join(output_path, "prompt2/ENtoPT.json")
    arquivo_saida = os.path.join(output_path, "prompt2/PTtoEN.json")

    anotacoes = []


    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        vetor = json.load(f)

    for objeto in vetor:

        prompt1 = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso"
        prompt2 = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso. A frase pode ou não conter metáfora"

        messages = [
            {"role": "user", "content": prompt2}
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

    args = parser.parse_args()

    main(args.model_id, args.hf_token, args.output_path)



