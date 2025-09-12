from huggingface_hub import login
import argparse
import transformers
import pandas as pd
import torch
import json
import os

# Arquivo pra rodar o modelo da meta, o marian e o gemmaX

def main(model_id, hf_token, output_path):
    login(token=hf_token)

    dataset_name = output_path.split("_", 1)[1].split("/")[0]

    if dataset_name == "newsmet":
        df = pd.read_csv("comparacao_datasets/newsmet.csv")
        term = "Text"
    elif dataset_name == "manual_data":
        df = pd.read_parquet("comparacao_datasets/manual_data.parquet")
        term = "Sentence"

    device = 0 if torch.cuda.is_available() else -1
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    pipeline = transformers.pipeline(
        # task="translation_EN_to_PT",
        task="translation",
        trust_remote_code=True,
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"dtype": torch.bfloat16},
        device=device,
        src_lang="eng_Latn",
        tgt_lang="por_Latn"
    )
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    os.makedirs(output_path, exist_ok=True)
    arquivo_saida = os.path.join(output_path, "/PTtoEN.json")
        
    anotacoes = []
    
    for frase in df[term]:

        message = ">>pt<<" + frase

        outputs = pipeline(
            message,
            max_new_tokens=200,
            do_sample=True,
            temperature=1,
            top_p=0.95,
        )

        generated_texts = outputs[0]['translation_text']

        result = {
            "fraseEN" : frase,        
            "traducaoPT": generated_texts
        }
        
        anotacoes.append(result)
        
        with open(arquivo_saida, "w", encoding="utf-8") as f:
            json.dump(anotacoes, f, ensure_ascii=False, indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text generation with Hugging Face pipeline.")
    parser.add_argument('--model_id', type=str, required=True, help="Hugging Face Model ID.")
    parser.add_argument('--hf_token', type=str, required=True, help="Hugging Face API token.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the generated outputs.")

    args = parser.parse_args()

    main(args.model_id, args.hf_token, args.output_path)