import pandas as pd
import json
import csv
from collections import defaultdict
import math


def calcular_medias(modelos, dataset):
    df = pd.DataFrame(columns=["Modelo", "Bleu_media","Rouge_media",  "Bertscore_media", "Bleurt_media", "Comet22_media", "Kiwi-XL_media", "XComet-XL_media"])
    for modelo in modelos:
        caminho = f"dataset_{dataset}/{modelo}/frases_traduzidas_com_metricas.json"
        with open(caminho, "r") as f:
            dados = json.load(f)
        
        object = {
            "Modelo": "",
            "Rouge_media": 0.0,
            "Bleu_media": 0.0,
            "Bertscore_media": 0.0,
            "Bleurt_media": 0.0,
            "Comet22_media": 0.0,
            "Kiwi-XL_media": 0.0,
            "XComet-XL_media": 0.0
        }
        
        qtd = len(dados)
    
        for objeto in dados:
            object['Rouge_media'] += objeto["ROUGE"]["rougeL"]
            object['Bleu_media'] += objeto["BLEU"]["bleu"]
            object["Bertscore_media"] += objeto["BERTSCORE"]["f1"]
            object["Bleurt_media"] += objeto["BLEURT"]["scores"]
            object["Comet22_media"] += objeto["COMET22"]["scores"]
            object["Kiwi-XL_media"] += objeto["KIWI-XL"]["scores"]
            object["XComet-XL_media"] += objeto["XCOMET-XL"]["scores"]

        object["Modelo"] = modelo.split("/")[0]
        object["Rouge_media"] = round(object["Rouge_media"] / qtd, 3)
        object["Bleu_media"] = round(object["Bleu_media"] / qtd, 3)
        object["Bertscore_media"] = round(object["Bertscore_media"] / qtd, 3)
        object["Bleurt_media"] = round(object["Bleurt_media"] / qtd, 3)
        object["Comet22_media"] = round(object["Comet22_media"] / qtd, 3)
        object["Kiwi-XL_media"] = round(object["Kiwi-XL_media"] / qtd, 3)
        object["XComet-XL_media"] = round(object["XComet-XL_media"] / qtd, 3)

        df.loc[len(df)] = object

    return df

def calcular_desvio_padrao(modelos, dataset, complemento):

    df = pd.DataFrame(columns=["Modelo", "Bleu_desvio","Rouge_desvio",  "Bertscore_desvio", "Bleurt_desvio", "Comet22_desvio", "Kiwi-XL_desvio", "XComet-XL_desvio"])

    tabela = ler_csv_por_coluna(f"dataset_{dataset}/[CSV] media_desvio_padrao/medias_{complemento}.csv")

    for i, modelo in enumerate(modelos):
        caminho = f"dataset_{dataset}/{modelo}/frases_traduzidas_com_metricas.json"
        with open(caminho, "r") as f:
            dados = json.load(f)

        qtd = len(dados)

        object = {
            "Modelo": "",
            "Bleu_desvio": 0.0,
            "Rouge_desvio": 0.0,
            "Bertscore_desvio": 0.0,
            "Bleurt_desvio": 0.0,
            "Comet22_desvio": 0.0,
            "Kiwi-XL_desvio": 0.0,
            "XComet-XL_desvio": 0.0
        }
    
        for objeto in dados:
            object['Bleu_desvio'] += ((objeto["BLEU"]["bleu"] - float(tabela["Bleu_media"][i])) ** 2)
            object['Rouge_desvio'] += ((objeto["ROUGE"]["rougeL"] - float(tabela["Rouge_media"][i])) ** 2)
            object["Bertscore_desvio"] += ((objeto["BERTSCORE"]["f1"] - float(tabela["Bertscore_media"][i])) ** 2)
            object["Bleurt_desvio"] += ((objeto["BLEURT"]["scores"] - float(tabela["Bleurt_media"][i])) ** 2)
            object["Comet22_desvio"] += ((objeto["COMET22"]["scores"] - float(tabela["Comet22_media"][i])) ** 2)
            object["Kiwi-XL_desvio"] += ((objeto["KIWI-XL"]["scores"] - float(tabela["Kiwi-XL_media"][i])) ** 2)
            object["XComet-XL_desvio"] += ((objeto["XCOMET-XL"]["scores"] - float(tabela["XComet-XL_media"][i])) ** 2)

        object["Modelo"] = modelo.split("/")[0]
        object["Rouge_desvio"] = round(math.sqrt(object["Rouge_desvio"] / qtd), 3)
        object["Bleu_desvio"] = round(math.sqrt(object["Bleu_desvio"] / qtd), 3)
        object["Bertscore_desvio"] = round(math.sqrt(object["Bertscore_desvio"] / qtd), 3)
        object["Bleurt_desvio"] = round(math.sqrt(object["Bleurt_desvio"] / qtd), 3)
        object["Comet22_desvio"] = round(math.sqrt(object["Comet22_desvio"] / qtd), 3)
        object["Kiwi-XL_desvio"] = round(math.sqrt(object["Kiwi-XL_desvio"] / qtd), 3)
        object["XComet-XL_desvio"] = round(math.sqrt(object["XComet-XL_desvio"] / qtd), 3)

        df.loc[len(df)] = object

    return df

def ler_csv_por_coluna(caminho_arquivo):
    colunas = defaultdict(list)
    with open(caminho_arquivo, mode='r', encoding='utf-8') as f:
        leitor = csv.DictReader(f)
        for linha in leitor:
            for chave, valor in linha.items():
                colunas[chave].append(valor)
    return dict(colunas)

if __name__ == "__main__":
    datasets = ["manualdata", "newsmet"]
    modelos = [
        ["gpt/prompt1/", "gemini/prompt1/", "gemma3/prompt1/", "llama/prompt1/", "mistral/prompt1/", "qwen/prompt1/"],
        ["gpt/prompt2/", "gemini/prompt2/", "gemma3/prompt2/",  "llama/prompt2/", "mistral/prompt2/","qwen/prompt2/"],
    ]
    # modelos_prompt1 = ["gpt/prompt1/", "gemini/prompt1/", "gemma3/prompt1/", "llama/prompt1/", "mistral/prompt1/", "qwen/prompt1/"]
    # modelos_prompt2 = ["gpt/prompt2/", "gemini/prompt2/", "gemma3/prompt2/",  "llama/prompt2/", "mistral/prompt2/","qwen/prompt2/"]
    modelos_tradicionais = ["gemmaX", "marian", "meta"]
    metricas = ["ROUGE/rougeL", "BLEU/bleu", "BERTSCORE/f1", "BLEURT/scores", "COMET22/scores", "KIWI-XL/scores", "XCOMET-XL/scores"]

    for dataset in datasets:
            for modelo in modelos:
                df = calcular_medias(modelo, dataset)
                df.to_csv(f"dataset_{dataset}/[CSV] media_desvio_padrao/medias_{modelo[0].split('/')[1]}.csv", index=False)

            df = calcular_medias(modelos_tradicionais, dataset)
            df.to_csv(f"dataset_{dataset}/[CSV] media_desvio_padrao/medias_tradicionais.csv", index=False)
    
            for modelo in modelos:
                df = calcular_desvio_padrao(modelo, dataset, modelo[0].split('/')[1])
                df.to_csv(f"dataset_{dataset}/[CSV] media_desvio_padrao/desvio_padrao_{modelo[0].split('/')[1]}.csv", index=False)

            df = calcular_desvio_padrao(modelos_tradicionais, dataset, modelo[0].split('/')[1])
            df.to_csv(f"dataset_{dataset}/[CSV] media_desvio_padrao/desvio_padrao_tradicionais.csv", index=False)