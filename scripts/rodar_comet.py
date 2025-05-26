import json
from comet import download_model, load_from_checkpoint
from huggingface_hub import login
import torch

login(token="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")

model_path = download_model("Unbabel/wmt22-comet-da")
kiwi_model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
xcomet_model_path = download_model("Unbabel/XCOMET-XL")

model = load_from_checkpoint(model_path)
model_kiwi = load_from_checkpoint(kiwi_model_path)
model_xcomet = load_from_checkpoint(xcomet_model_path)

# dataset = ["dataset_newsmet/", "dataset_manueladata/"]
# pasta1 =["gemma3/prompt1/", "gemma3/prompt2/", "gpt/prompt1/", "gpt/prompt2/", "llama/promp1/", "llama/prompt2/", "marian/", "meta/", "mistral/prompt1/", "mistral/prompt2/", "qwen/prompt1/", "qwen/prompt2/"]
datasets = ["dataset_newsmet/"]
pastas = ["gemmaX/"]

for dataset in datasets:
    for pasta in pastas:
        with open(f'{dataset}{pasta}frases_traduzidas_com_metricas.json', 'r', encoding='utf-8') as file:
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
            print(f"Frase {i}. {dataset}{pasta}")

        with open(f'{dataset}{pasta}frases_traduzidas_com_metricas.json', 'w', encoding='utf-8') as file:
            json.dump(dados, file, ensure_ascii=False, indent=4)
