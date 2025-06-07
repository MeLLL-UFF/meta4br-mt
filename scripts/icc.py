import pingouin as pg
import pandas as pd
import json

# Um pra prompt1 e tradicionais e um pra prompt2 e tradicionais
# ICC1
# Fazer um geral pra todos os modelos, seja prompt1, prompt2 e tradicionais

def montar_df(modelos, dataset, df, object, metricas, frases):
    dados_modelos = {}
    
    # Lê todos os arquivos de uma vez e armazena em memória
    for modelo in modelos:
        caminho = f"dataset_{dataset}/{modelo}/frases_traduzidas_com_metricas.json"
        with open(caminho, "r") as f:
            dados_modelos[modelo] = json.load(f)
    
    for i, frase in enumerate(frases):
        print(f"Frase {i+1} de {len(frases)}")
        for modelo in modelos:
            dados = dados_modelos[modelo]
            if i >= len(dados):
                continue
            for metrica in metricas:
                nome_metrica, chave = metrica.split("/")
                object["frases_ingles"].append(frase)
                object["metrica"].append(nome_metrica)
                object["modelo"].append(modelo.split("/")[0])
                object["score"].append(dados[i][nome_metrica][chave])
    
    df = pd.DataFrame(object)
    print(df.head())
    return df
        

if __name__ == "__main__":
    datasets = ["manualdata", "newsmet"]
    modelos_prompt1 = ["gemini/prompt1/", "gemma3/prompt1/", "gemmaX", "gpt/prompt1/", "llama/prompt1/", "marian", "meta", "mistral/prompt1/", "qwen/prompt1/"]
    modelos_prompt2 = ["gemini/prompt2/", "gemma3/prompt2/", "gemmaX", "gpt/prompt2/", "llama/prompt2/", "marian", "meta", "mistral/prompt2/","qwen/prompt2/"]
    todos_modelos = ["gemini/prompt1/", "gemma3/prompt1/", "gpt/prompt1/", "llama/prompt1/", "mistral/prompt1/", "qwen/prompt1/", "gemini/prompt2/", "gemma3/prompt2/", "gpt/prompt2/", "llama/prompt2/", "mistral/prompt2/","qwen/prompt2/", "gemmaX", "marian", "meta"]
    vet_manualdata = pd.read_parquet("comparacao_datasets/manual_data.parquet")["Sentence"].tolist()
    vet_newsmet = pd.read_csv("comparacao_datasets/newsmet.csv")["Text"].tolist()

    metricas = ["ROUGE/rougeL", "BLEU/bleu", "BERTSCORE/f1", "BLEURT/scores", "COMET22/scores", "KIWI-XL/scores", "XCOMET-XL/scores"]
    for dataset in datasets:
        object = {
            "frases_ingles": [],
            "metrica": [],
            "modelo": [],
            "score": [],
        }

        df = pd.DataFrame()
        if dataset == "manualdata":
            df = montar_df(modelos_prompt1, dataset, df, object, metricas, vet_manualdata)
            df.to_csv(f"dataset_{dataset}/df_prompt1.csv", index=False)
            df = montar_df(modelos_prompt2, dataset, df, object, metricas, vet_manualdata)
            df.to_csv(f"dataset_{dataset}/df_prompt2.csv", index=False)
            df = montar_df(todos_modelos, dataset, df, object, metricas, vet_manualdata)
            df.to_csv(f"dataset_{dataset}/df_geral.csv", index=False)
        elif dataset == "newsmet":
            df = montar_df(modelos_prompt1, dataset, df, object, metricas, vet_newsmet)
            df.to_csv(f"dataset_{dataset}/df_prompt1.csv", index=False)
            df = montar_df(modelos_prompt2, dataset, df, object, metricas, vet_newsmet)
            df.to_csv(f"dataset_{dataset}/df_prompt2.csv", index=False)
            df = montar_df(todos_modelos, dataset, df, object, metricas, vet_newsmet)
            df.to_csv(f"dataset_{dataset}/df_geral.csv", index=False)

    df_manualdata1 = pd.read_csv("dataset_manualdata/df_prompt1.csv")
    df_manualdata2 = pd.read_csv("dataset_manualdata/df_prompt2.csv")
    df_manualdata_geral = pd.read_csv("dataset_manualdata/df_geral.csv")

    df_newsmet1 = pd.read_csv("dataset_newsmet/df_prompt1.csv")
    df_newsmet2 = pd.read_csv("dataset_newsmet/df_prompt2.csv")
    df_newsmet_geral = pd.read_csv("dataset_newsmet/df_geral.csv")


    icc_manualdata1 = pg.intraclass_corr(data=df_manualdata1, targets='frases_ingles', raters='modelo', ratings='score').round(3)
    icc_manualdata1.to_csv("dataset_manualdata/icc_final_prompt1.csv", index=False)
    icc_manualdata2 = pg.intraclass_corr(data=df_manualdata2, targets='frases_ingles', raters='modelo', ratings='score').round(3)
    icc_manualdata2.to_csv("dataset_manualdata/icc_final_prompt2.csv", index=False)
    icc_manualdata_geral = pg.intraclass_corr(data=df_manualdata_geral, targets='frases_ingles', raters='modelo', ratings='score').round(3)
    icc_manualdata_geral.to_csv("dataset_manualdata/icc_final_geral.csv", index=False)

    icc_newsmet1 = pg.intraclass_corr(data=df_newsmet1, targets='frases_ingles', raters='modelo', ratings='score').round(3)
    icc_newsmet1.to_csv("dataset_newsmet/icc_final_prompt1.csv", index=False)
    icc_newsmet2 = pg.intraclass_corr(data=df_newsmet2, targets='frases_ingles', raters='modelo', ratings='score').round(3)
    icc_newsmet2.to_csv("dataset_newsmet/icc_final_prompt2.csv", index=False)
    icc_newsmet_geral = pg.intraclass_corr(data=df_newsmet_geral, targets='frases_ingles', raters='modelo', ratings='score').round(3)
    icc_newsmet_geral.to_csv("dataset_newsmet/icc_final_geral.csv", index=False)

# Então, no seu caso, a resposta depende de o que você quer medir com o ICC:
# 🔸 Se você quer avaliar consistência entre modelos:
# targets = frase_id (ou a frase em inglês)
# raters = modelo
# ratings = score (por exemplo: ROUGE-L)
# Aqui, você mede se os modelos concordam entre si ao avaliar uma mesma frase segundo uma métrica específica.
# 🔸 Se você quiser avaliar consistência entre métricas (menos comum nesse caso):
# targets = frase_id
# raters = metrica
# ratings = score
# Isso mediria se as métricas concordam entre si ao avaliar uma mesma tradução.