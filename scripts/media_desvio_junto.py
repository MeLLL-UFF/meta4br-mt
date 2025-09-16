import pandas as pd
import json
import csv
from collections import defaultdict
import math


def calcular_medias(modelos, dataset, prompt):
    df = pd.DataFrame(columns=["Modelo", "Bleu_media", "Rouge_media", "Bertscore_media", "Bleurt_media", "Comet22_media", "Kiwi-XL_media", "XComet-XL_media"])
    for modelo in modelos:
        caminho = f"dataset_{dataset}/{modelo}/{prompt}frases_traduzidas_com_metricas.json"
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

def calcular_desvio_padrao(modelos, dataset, prompt):

    df = pd.DataFrame(columns=["Modelo", "Bleu_desvio","Rouge_desvio",  "Bertscore_desvio", "Bleurt_desvio", "Comet22_desvio", "Kiwi-XL_desvio", "XComet-XL_desvio"])

    tabela = ler_csv_por_coluna(f"dataset_{dataset}/[CSV] media_desvio_padrao/medias_{prompt}.csv")
    
    for i, modelo in enumerate(modelos):
        
        if prompt == "tradicionais":
            caminho = f"dataset_{dataset}/{modelo}/frases_traduzidas_com_metricas.json"
        else:
            caminho = f"dataset_{dataset}/{modelo}/{prompt}/frases_traduzidas_com_metricas.json"

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

    modelos = ["gemini", "gemma3", "gpt", "llama", "mistral", "qwen"]
    modelos_traducao = ["gemmaX", "marian", "meta"]

    metricas = ["ROUGE/rougeL", "BLEU/bleu", "BERTSCORE/f1", "BLEURT/scores", "COMET22/scores", "KIWI-XL/scores", "XCOMET-XL/scores"]

    for dataset in ["manual_data", "newsmet"]:
        
        # Modelos de tradução não possuem vários prompts
        df = calcular_medias(modelos_traducao, dataset, "")
        df.to_csv(f"dataset_{dataset}/[CSV] media_desvio_padrao/medias_tradicionais.csv", index=False)

        df = calcular_desvio_padrao(modelos_traducao, dataset, "tradicionais")
        df.to_csv(f"dataset_{dataset}/[CSV] media_desvio_padrao/desvio_padrao_tradicionais.csv", index=False)

        for prompt in ["prompt1", "prompt2", "prompt3", "prompt4"]:
            
            df = calcular_medias(modelos, dataset, prompt + "/")
            df.to_csv(f"dataset_{dataset}/[CSV] media_desvio_padrao/medias_{prompt}.csv", index=False)

            df = calcular_desvio_padrao(modelos, dataset, prompt)
            df.to_csv(f"dataset_{dataset}/[CSV] media_desvio_padrao/desvio_padrao_{prompt}.csv", index=False)