import json
from comet import download_model, load_from_checkpoint
import torch

# model_path = download_model("Unbabel/wmt22-comet-da")
kiwi_model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
# xcomet_model_path = download_model("Unbabel/XCOMET-XXL")

# model = load_from_checkpoint(model_path)
model_kiwi = load_from_checkpoint(kiwi_model_path)
# model_xcomet = load_from_checkpoint(xcomet_model_path)

with open('dataset_manualdata/gemini/prompt1/frases_traduzidas_com_metricas.json', 'r', encoding='utf-8') as file:
    dados = json.load(file)

comet_data = [
    {
        "src": obj["portugues_traduzido"],
        "mt": obj["ingles_traduzido"],
        "ref": obj["ingles_original"],
    }
    for obj in dados
]

# results = model.predict(comet_data)
# torch.cuda.empty_cache()

results_kiwi = model_kiwi.predict(comet_data, batch_size=8)
torch.cuda.empty_cache()

# results_xcomet = model_xcomet.predict(comet_data, batch_size=1)
# torch.cuda.empty_cache()

for i in range(len(dados)):
    # dados[i]["COMET22"] = {"scores": results.scores[i]}
    dados[i]["KIWI-XXL"] = {"scores": results_kiwi.scores[i]}
    # dados[i]["XCOMET"] = {"scores": results_xcomet.scores[i]}
    print(f"Frase {i}.")

with open('dataset_manualdata/gemini/prompt2/frases_traduzidas_com_metricas.json', 'w', encoding='utf-8') as file:
    json.dump(dados, file, ensure_ascii=False, indent=4)
