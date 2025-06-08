import pandas as pd
import json
import os


def carregar_dados(modelos, dataset):
    todos_dados = []
    for modelo in modelos:
        caminho = f"dataset_{dataset}/{modelo}/frases_traduzidas_com_metricas.json"
        with open(caminho, "r") as f:
            dados_json = json.load(f)
        for d in dados_json:
            dados_flat = {
                "Modelo": modelo.split("/")[0],
                "BLEU": d["BLEU"]["bleu"],
                "ROUGE": d["ROUGE"]["rougeL"],
                "BERTSCORE": d["BERTSCORE"]["f1"],
                "BLEURT": d["BLEURT"]["scores"],
                "COMET22": d["COMET22"]["scores"],
                "KIWI-XL": d["KIWI-XL"]["scores"],
                "XCOMET-XL": d["XCOMET-XL"]["scores"]
            }
            todos_dados.append(dados_flat)
    return pd.DataFrame(todos_dados)


def calcular_estatisticas(df):
    medias = df.groupby("Modelo").mean(numeric_only=True).round(3)
    desvios = df.groupby("Modelo").std(numeric_only=True).round(3)

    medias = medias.rename(columns={
        "BLEU": "Bleu_media",
        "ROUGE": "Rouge_media",
        "BERTSCORE": "Bertscore_media",
        "BLEURT": "Bleurt_media",
        "COMET22": "Comet22_media",
        "KIWI-XL": "Kiwi-XL_media",
        "XCOMET-XL": "XComet-XL_media"
    }).reset_index()

    desvios = desvios.rename(columns={
        "BLEU": "Bleu_desvio",
        "ROUGE": "Rouge_desvio",
        "BERTSCORE": "Bertscore_desvio",
        "BLEURT": "Bleurt_desvio",
        "COMET22": "Comet22_desvio",
        "KIWI-XL": "Kiwi-XL_desvio",
        "XCOMET-XL": "XComet-XL_desvio"
    }).reset_index()

    return medias, desvios


if __name__ == "__main__":
    datasets = ["manualdata", "newsmet"]
    modelos = [
        ["gpt/prompt1/", "gemini/prompt1/", "gemma3/prompt1/", "llama/prompt1/", "mistral/prompt1/", "qwen/prompt1/"],
        ["gpt/prompt2/", "gemini/prompt2/", "gemma3/prompt2/", "llama/prompt2/", "mistral/prompt2/", "qwen/prompt2/"]
    ]
    modelos_tradicionais = ["gemmaX", "marian", "meta"]

    for dataset in datasets:
        for grupo_modelos in modelos:
            df = carregar_dados(grupo_modelos, dataset)
            medias, desvios = calcular_estatisticas(df)

            prompt_nome = grupo_modelos[0].split("/")[1]
            pasta = f"dataset_{dataset}/[CSV] media_desvio_padrao"
            os.makedirs(pasta, exist_ok=True)

            medias.to_csv(f"{pasta}/medias_{prompt_nome}.csv", index=False)
            desvios.to_csv(f"{pasta}/desvio_padrao_{prompt_nome}.csv", index=False)

        df_trad = carregar_dados(modelos_tradicionais, dataset)
        medias, desvios = calcular_estatisticas(df_trad)
        medias.to_csv(f"dataset_{dataset}/[CSV] media_desvio_padrao/medias_tradicionais.csv", index=False)
        desvios.to_csv(f"dataset_{dataset}/[CSV] media_desvio_padrao/desvio_padrao_tradicionais.csv", index=False)
