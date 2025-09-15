import json
from comet import download_model, load_from_checkpoint
from huggingface_hub import login
import torch
import os

def rodar_comet(dataset_name, model_name, prompt_id):

    model_path = download_model("Unbabel/wmt22-comet-da")
    kiwi_model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    xcomet_model_path = download_model("Unbabel/XCOMET-XL")

    model = load_from_checkpoint(model_path)
    model_kiwi = load_from_checkpoint(kiwi_model_path)
    model_xcomet = load_from_checkpoint(xcomet_model_path)

    with open(f'dataset_{dataset_name}/{model_name}/{prompt_id}frases_traduzidas_com_metricas.json', 'r', encoding='utf-8') as file:
        dados = json.load(file)

    comet_data = [
        {
            "src": obj["portugues_traduzido"],
            "mt": obj["ingles_traduzido"],
            "ref": obj["ingles_original"],
        }
        for obj in dados
    ]

    results = model.predict(comet_data)
    torch.cuda.empty_cache()

    results_kiwi = model_kiwi.predict(comet_data)
    torch.cuda.empty_cache()

    results_xcomet = model_xcomet.predict(comet_data)
    torch.cuda.empty_cache()

    for i in range(len(dados)):
        dados[i]["COMET22"] = {"scores": results.scores[i]}
        dados[i]["KIWI-XL"] = {"scores": results_kiwi.scores[i]}
        dados[i]["XCOMET-XL"] = {"scores": results_xcomet.scores[i]}

    with open(f'dataset_{dataset_name}/{model_name}/{prompt_id}frases_traduzidas_com_metricas.json', 'w', encoding='utf-8') as file:
        json.dump(dados, file, ensure_ascii=False, indent=4)

def teste(dataset_name, model_name, prompt_id):
    import json

dataset_name = "newsmet"
model_name = "gemini"
prompt_id = "prompt3/"

# Carrega o arquivo
with open(f'dataset_{dataset_name}/{model_name}/{prompt_id}frases_traduzidas_com_metricas.json', 'r', encoding='utf-8') as file:
    dados = json.load(file)

# Verifica quais estão com None
for i, obj in enumerate(dados):
    problemas = []
    if obj.get("portugues_traduzido") is None:
        problemas.append("portugues_traduzido")
    if obj.get("ingles_traduzido") is None:
        problemas.append("ingles_traduzido")
    if obj.get("ingles_original") is None:
        problemas.append("ingles_original")
    
    if problemas:
        print(f"Objeto {i} tem None nos campos: {problemas}")


if __name__ == "__main__":

    api_key = os.environ.get("HF_TOKEN")

    if not api_key :
        print("Variável de ambiente HF_TOKEN não encontrada.")

    for dataset in ["manual_data", "newsmet"]:
            for model in ["gemini", "gemma3", "gpt", "llama", "mistral", "qwen", "gemmaX", "marian", "meta"]:
                for prompt in ["prompt1", "prompt2", "prompt3", "prompt4"]:

                    # Os modelos de tradução, que não possuem prompt
                    if model in ["gemmaX", "marian", "meta"]: 
                        if prompt == "prompt2": # Só roda 1 vez para os modelos de tradução
                            break

                        prompt = ""

                    else :
                        prompt = prompt + "/"

                    print(f'Rodando: dataset_{dataset}/{model}/{prompt}frases_traduzidas_com_metricas.json')
                    rodar_comet(dataset, model, prompt)
                    teste(dataset, model, prompt)
                    print(f'Arquivo salvo em dataset_{dataset}/{model}/{prompt}frases_traduzidas_com_metricas.json')

