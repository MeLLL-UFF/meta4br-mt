import json
import pandas as pd


def carregar_labels(dataset_name):
    if dataset_name == "manual_data":
        df = pd.read_parquet("comparacao_datasets/manual_data.parquet")
        return {row["Sentence"]: int(row["Label"]) for _, row in df.iterrows()}

    if dataset_name == "newsmet":
        df = pd.read_csv("comparacao_datasets/newsmet.csv")
        return {
            row["Text"]: 1 if row["predicted_label"] == "metaphorical" else 0
            for _, row in df.iterrows()
        }

    return {}

def juntar_jsons(dataset_name, model_name, prompt_id):

    with open(f'dataset_{dataset_name}/{model_name}/{prompt_id}ENtoPT.json', 'r', encoding='utf-8') as file:
        data1 = json.load(file)
    with open(f'dataset_{dataset_name}/{model_name}/{prompt_id}PTtoEN.json', 'r', encoding='utf-8') as file:
        data2 = json.load(file)

    labels = carregar_labels(dataset_name)

    new_data = []
    indice = 0
    for item in data1:
        result = {
            "ingles_original": item["fraseEN"],
            "portugues_traduzido": item["traducaoPT"],
            "ingles_traduzido": data2[indice]["traducaoEN"],
            "label": labels.get(item["fraseEN"])
        }

        new_data.append(result)
        result = {}
        indice += 1

    with open(f'dataset_{dataset_name}/{model_name}{prompt_id}/frases_traduzidas.json', 'w', encoding='utf-8') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)

    print(f'Arquivos juntados e salvos em dataset_{dataset_name}/{model_name}{prompt_id}/frases_traduzidas.json')

# Se não quer juntar de uma vez pra todos os datasets, todos os modelos e todos os prompts (foi fazendo pro partes), deve alterar os FORs abaixo)
if __name__ == "__main__":
    for dataset in ["manual_data", "newsmet"]:
        for model in ["gemini", "gemma3", "gpt", "llama", "mistral", "qwen", "gemmaX", "marian", "meta"]:
            for prompt in ["prompt1", "prompt2", "prompt3", "prompt4"]:
                if model in ["gemmaX", "marian", "meta"]: # Os modelos de tradução, que não possuem prompt
                    if prompt == "prompt2": # Só roda 1 vez para os modelos de tradução
                        break

                    prompt = ""

                prompt = prompt + "/"
                juntar_jsons(dataset, model, prompt)