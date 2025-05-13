import json
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

with open('dataset_newsmet/qwen/prompt2/frases_traduzidas_com_metricas.json', 'r', encoding='utf-8') as file:
    dados = json.load(file)

    comet_data = []
    for objeto in dados:
        comet_data.append({
            "src": objeto["portugues_traduzido"],
            "mt": objeto["ingles_traduzido"],
            "ref": objeto["ingles_original"],
        })


    results = model.predict(comet_data, batch_size=8, gpus=1)
    scores = results.scores

    for i in range(len(dados)):
        dados[i]["COMET"] = {"scores" : scores[i]}

    with open('dataset_newsmet/qwen/prompt2/frases_traduzidas_com_metricas.json', 'w', encoding='utf-8') as file:
        json.dump(dados, file, ensure_ascii=False, indent=4)
