import pandas as pd
import json
import os


def carregar_dados(modelos, dataset, frases, prompt):
    todos_dados_com_metafora = []
    todos_dados_sem_metafora = []
    for modelo in modelos:
        caminho = f"dataset_{dataset}/{modelo}/{prompt}frases_traduzidas_com_metricas.json"
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

            # Se está em frases é porque é metafórica, senão está é porque não é
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

    modelos = ["gemini", "gemma3", "gpt", "llama", "mistral", "qwen"]
    modelos_traducao = ["gemmaX", "marian", "meta"]

    for dataset in ["manual_data", "newsmet"]:

        pasta = f"dataset_{dataset}/[CSV] media_desvio_padrao/separado/"
        os.makedirs(pasta, exist_ok=True)

        if dataset == "newsmet":
            df_original = pd.read_csv(f"comparacao_datasets/newsmet.csv")
            frases = list(set(df_original.loc[df_original["predicted_label"] == "metaphorical", "Text"]))

        elif dataset == "manual_data":
            df_original = pd.read_parquet(f"comparacao_datasets/manual_data.parquet")
            frases = list(set(df_original.loc[df_original["Label"] == 1, "Sentence"]))


        #Modelos de tradução não têm diferentes prompts
        df_trad_com_metafora, df_trad_sem_metafora = carregar_dados(modelos_traducao, dataset, frases, "")

        medias, desvios = calcular_estatisticas(df_trad_com_metafora)
        medias.to_csv(f"{pasta}medias_tradicionais_com_metafora.csv", index=False)
        desvios.to_csv(f"{pasta}desvio_padrao_tradicionais_com_metafora.csv", index=False)
        print(f"CSV salvo em {pasta}*_tradicionais_com_metafora.csv")

        medias, desvios = calcular_estatisticas(df_trad_sem_metafora)
        medias.to_csv(f"{pasta}medias_tradicionais_sem_metafora.csv", index=False)
        desvios.to_csv(f"{pasta}desvio_padrao_tradicionais_sem_metafora.csv", index=False)
        print(f"CSV salvo em {pasta}*_tradicionais_sem_metafora.csv")

        for prompt in ["prompt3", "prompt4"]:

            df_com_metafora, df_sem_metafora = carregar_dados(modelos, dataset, frases, prompt + "/") # "/" só pra ajudar nos nomes das pastas nas funções
            medias, desvios = calcular_estatisticas(df_com_metafora)
            medias.to_csv(f"{pasta}medias_{prompt}_com_metafora.csv", index=False)
            desvios.to_csv(f"{pasta}desvio_padrao_{prompt}_com_metafora.csv", index=False)
            print(f"CSV salvo em {pasta}*_{prompt}_com_metafora.csv")

            medias, desvios = calcular_estatisticas(df_sem_metafora)
            medias.to_csv(f"{pasta}medias_{prompt}_sem_metafora.csv", index=False)
            desvios.to_csv(f"{pasta}desvio_padrao_{prompt}_sem_metafora.csv", index=False)
            print(f"CSV salvo em {pasta}*_{prompt}_sem_metafora.csv")



                
