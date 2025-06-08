import pandas as pd
import json

def calcular_medias(modelos, dataset, metricas, frases, df):
    # Lê todos os arquivos de uma vez e armazena em memória
    qtd = len(modelos)
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

if __name__ == "__main__":
    datasets = ["manualdata", "newsmet"]
    modelos_prompt1 = ["gemini/prompt1/", "gemma3/prompt1/", "gpt/prompt1/", "llama/prompt1/", "mistral/prompt1/", "qwen/prompt1/"]
    modelos_prompt2 = ["gemini/prompt2/", "gemma3/prompt2/", "gpt/prompt2/", "llama/prompt2/", "mistral/prompt2/","qwen/prompt2/"]
    modelos_tradicionais = ["gemmaX", "marian", "meta"]
    metricas = ["ROUGE/rougeL", "BLEU/bleu", "BERTSCORE/f1", "BLEURT/scores", "COMET22/scores", "KIWI-XL/scores", "XCOMET-XL/scores"]
    vet_manualdata = pd.read_parquet("comparacao_datasets/manual_data.parquet")["Sentence"].tolist()
    vet_newsmet = pd.read_csv("comparacao_datasets/newsmet.csv")["Text"].tolist()

    for dataset in datasets:
        df = pd.DataFrame(columns=["Modelo", "Rouge_media", "Bleu_media", "Bertscore_media", "Bleurt_media", "Comet22_media", "Kiwi-XL_media", "XComet-XL_media"])
        if dataset == "manualdata":
            df = calcular_medias(modelos_prompt1, dataset, metricas, vet_manualdata, df)
            df.to_csv(f"dataset_{dataset}/medias_prompt1.csv", index=False)
            df = pd.DataFrame(columns=["Modelo", "Rouge_media", "Bleu_media", "Bertscore_media", "Bleurt_media", "Comet22_media", "Kiwi-XL_media", "XComet-XL_media"])

            df = calcular_medias(modelos_prompt2, dataset, metricas, vet_manualdata, df)
            df.to_csv(f"dataset_{dataset}/medias_prompt2.csv", index=False)
            df = pd.DataFrame(columns=["Modelo", "Rouge_media", "Bleu_media", "Bertscore_media", "Bleurt_media", "Comet22_media", "Kiwi-XL_media", "XComet-XL_media"])

            df = calcular_medias(modelos_tradicionais, dataset, metricas, vet_manualdata, df)
            df.to_csv(f"dataset_{dataset}/medias_tradicionais.csv", index=False)
            df = pd.DataFrame(columns=["Modelo", "Rouge_media", "Bleu_media", "Bertscore_media", "Bleurt_media", "Comet22_media", "Kiwi-XL_media", "XComet-XL_media"])

        elif dataset == "newsmet":
            df = calcular_medias(modelos_prompt1, dataset, metricas, vet_newsmet, df)
            df.to_csv(f"dataset_{dataset}/medias_prompt1.csv", index=False)
            df = pd.DataFrame(columns=["Modelo", "Rouge_media", "Bleu_media", "Bertscore_media", "Bleurt_media", "Comet22_media", "Kiwi-XL_media", "XComet-XL_media"])

            df = calcular_medias(modelos_prompt2, dataset, metricas, vet_newsmet, df)
            df.to_csv(f"dataset_{dataset}/medias_prompt2.csv", index=False)
            df = pd.DataFrame(columns=["Modelo", "_media", "Bleu_media", "Bertscore_media", "Bleurt_media", "Comet22_media", "Kiwi-XL_media", "XComet-XL_media"])

            df = calcular_medias(modelos_tradicionais, dataset, metricas, vet_newsmet, df)
            df.to_csv(f"dataset_{dataset}/medias_tradicionais.csv", index=False)
            df = pd.DataFrame(columns=["Modelo", "Rouge_media", "Bleu_media", "Bertscore_media", "Bleurt_media", "Comet22_media", "Kiwi-XL_media", "XComet-XL_media"])