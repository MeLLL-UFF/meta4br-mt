from huggingface_hub import login
import argparse
import transformers
import pandas as pd
import torch
import json
import gc
import os

# Arquivo pra rodar o modelo da meta e o marian

def main(model_id, hf_token, output_path):
    login(token=hf_token)

    device = f'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    pipeline = transformers.pipeline(
        task="translation",
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
        tgt_lang="eng_Latn",
        src_lang="por_Latn"
    )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    os.makedirs(output_path, exist_ok=True)
    arquivo_entrada = os.path.join(output_path, "ENtoPT.json")
    arquivo_saida = os.path.join(output_path, "PTtoEN.json")

    anotacoes = []

    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        vetor = json.load(f)

    for objeto in vetor:
        frase = objeto["traducaoPT"]

        # Tradução de português (por_Latn) para inglês (eng_Latn)
        outputs = pipeline(
            frase,
            max_new_tokens=2000,
            do_sample=True,
            temperature=1,
            top_p=0.95,
            src_lang="por_Latn",
            tgt_lang="eng_Latn"
        )

        traducao = outputs[0]["translation_text"]

        result = {
            "frasePT": frase,
            "traducaoEN": traducao
        }

        anotacoes.append(result)

        torch.cuda.empty_cache()
        gc.collect()

        with open(arquivo_saida, 'w', encoding='utf-8') as f:
            json.dump(anotacoes, f, ensure_ascii=False, indent=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run translation with Hugging Face pipeline.")
    parser.add_argument('--model_id', type=str, required=True, help="Hugging Face Model ID.")
    parser.add_argument('--hf_token', type=str, required=True, help="Hugging Face API token.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the generated outputs.")

    args = parser.parse_args()

    main(args.model_id, args.hf_token, args.output_path)
