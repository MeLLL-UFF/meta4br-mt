import pandas as pd
import json
import os


def carregar_dados(modelos, dataset, frases):
    todos_dados_com_metafora = []
    todos_dados_sem_metafora = []
    for modelo in modelos:
        caminho = f"dataset_{dataset}/{modelo}/frases_traduzidas_com_metricas.json"
        with open(caminho, "r") as f:
            dados_json = json.load(f)
        for d in dados_json:
            dados_metricas = {
                "frase": d["ingles_original"],
                "Modelo": modelo.split("/")[0],
                "BLEU": d["BLEU"]["bleu"],
                "ROUGE": d["ROUGE"]["rougeL"],
                "BERTSCORE": d["BERTSCORE"]["f1"],
                "BLEURT": d["BLEURT"]["scores"],
                "COMET22": d["COMET22"]["scores"],
                "KIWI-XL": d["KIWI-XL"]["scores"],
                "XCOMET-XL": d["XCOMET-XL"]["scores"]
            }

            if(dados_metricas["frase"] in frases): todos_dados_com_metafora.append(dados_metricas)
            else : todos_dados_sem_metafora.append(dados_metricas)
    return pd.DataFrame(todos_dados_com_metafora), pd.DataFrame(todos_dados_sem_metafora)


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
        if dataset == "newsmet":
            df_original = pd.read_csv(f"comparacao_datasets/{dataset}.csv")
            frases = list(set(df_original.loc[df_original["predicted_label"] == "metaphorical", "Text"]))

        elif dataset == "manualdata":
            df_original = pd.read_parquet(f"comparacao_datasets/manual_data.parquet")
            frases = list(set(df_original.loc[df_original["Label"] == 1, "Sentence"]))

        for grupo_modelos in modelos:
            prompt_nome = grupo_modelos[0].split("/")[1]
            pasta = f"dataset_{dataset}/[CSV] media_desvio_padrao"
            os.makedirs(pasta, exist_ok=True)

            df_com_metafora, df_sem_metafora = carregar_dados(grupo_modelos, dataset, frases)

            medias, desvios = calcular_estatisticas(df_com_metafora)
            medias.to_csv(f"{pasta}/separado/medias_{prompt_nome}_com_metafora.csv", index=False)
            desvios.to_csv(f"{pasta}/separado/desvio_padrao_{prompt_nome}_com_metafora.csv", index=False)

            medias, desvios = calcular_estatisticas(df_sem_metafora)
            medias.to_csv(f"{pasta}/separado/medias_{prompt_nome}_sem_metafora.csv", index=False)
            desvios.to_csv(f"{pasta}/separado/desvio_padrao_{prompt_nome}_sem_metafora.csv", index=False)

        df_trad_com_metafora, df_trad_sem_metafora = carregar_dados(modelos_tradicionais, dataset, frases)

        medias, desvios = calcular_estatisticas(df_trad_com_metafora)
        medias.to_csv(f"dataset_{dataset}/[CSV] media_desvio_padrao/separado/medias_tradicionais_com_metafora.csv", index=False)
        desvios.to_csv(f"dataset_{dataset}/[CSV] media_desvio_padrao/separado/desvio_padrao_tradicionais_com_metafora.csv", index=False)

        medias, desvios = calcular_estatisticas(df_trad_sem_metafora)
        medias.to_csv(f"dataset_{dataset}/[CSV] media_desvio_padrao/separado/medias_tradicionais_sem_metafora.csv", index=False)
        desvios.to_csv(f"dataset_{dataset}/[CSV] media_desvio_padrao/separado/desvio_padrao_tradicionais_sem_metafora.csv", index=False)
