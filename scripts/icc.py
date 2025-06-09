import pingouin as pg
import pandas as pd
import json
import os

# Um pra prompt1 e tradicionais e um pra prompt2 e tradicionais
# ICC1
# Fazer um geral pra todos os modelos, seja prompt1, prompt2 e tradicionais

def montar_df(modelos, dataset, object, metricas, frases):
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
    return df
        
def gerar_icc(dataset, complemento, metrica):
    path = f"dataset_{dataset}/[CSV] calculo_icc/df_{complemento}.csv"
    if not os.path.exists(path):
        print(f"Arquivo {path} não encontrado. Pulando ICC para {complemento} - {metrica}.")
    df = pd.read_csv(path)

    df_metrica_atual = df[df["metrica"] == metrica]
    print(df_metrica_atual.head())

    icc = pg.intraclass_corr(data=df_metrica_atual, targets='frases_ingles', raters='modelo', ratings='score').round(3)
    print(icc)
    icc.to_csv(f"dataset_{dataset}/[CSV] calculo_icc/icc_{complemento}/icc_{metrica}_{complemento}.csv", index=False)



if __name__ == "__main__":
    datasets = ["manualdata", "newsmet"]
    modelos = [
        ["gemini/prompt1/", "gemma3/prompt1/", "gemmaX", "gpt/prompt1/", "llama/prompt1/", "marian", "meta", "mistral/prompt1/", "qwen/prompt1/"],
        ["gemini/prompt2/", "gemma3/prompt2/", "gemmaX", "gpt/prompt2/", "llama/prompt2/", "marian", "meta", "mistral/prompt2/","qwen/prompt2/"]
    ]
    modelos_gerais = ["gemini/prompt1/", "gemma3/prompt1/", "gpt/prompt1/", "llama/prompt1/", "mistral/prompt1/", "qwen/prompt1/", "gemini/prompt2/", "gemma3/prompt2/",  "gpt/prompt2/", "llama/prompt2/",  "mistral/prompt2/", "qwen/prompt2/","gemmaX", "marian", "meta",]

    vet_manualdata = pd.read_parquet("comparacao_datasets/manual_data.parquet")["Sentence"].tolist()
    vet_newsmet = pd.read_csv("comparacao_datasets/newsmet.csv")["Text"].tolist()

    metricas = ["ROUGE/rougeL", "BLEU/bleu", "BERTSCORE/f1", "BLEURT/scores", "COMET22/scores", "KIWI-XL/scores", "XCOMET-XL/scores"]
    for dataset in datasets:

        df = pd.DataFrame()
        if dataset == "manualdata":
            for modelo in modelos:
                object = {
                    "frases_ingles": [],
                    "metrica": [],
                    "modelo": [],
                    "score": [],
                }
                df = montar_df(modelo, dataset, object, metricas, vet_manualdata)
                df.to_csv(f"dataset_{dataset}/[CSV] calculo_icc/df_{modelo[0].split("/")[1]}.csv", index=False)
                
            object = {
                "frases_ingles": [],
                "metrica": [],
                "modelo": [],
                "score": [],
            }
            df = montar_df(modelos_gerais, dataset, object, metricas, vet_manualdata)
            df.to_csv(f"dataset_{dataset}/[CSV] calculo_icc/df_geral.csv", index=False)
        elif dataset == "newsmet":
            for modelo in modelos:
                object = {
                    "frases_ingles": [],
                    "metrica": [],
                    "modelo": [],
                    "score": [],
                }
                df = montar_df(modelo, dataset, object, metricas, vet_newsmet)
                df.to_csv(f"dataset_{dataset}/[CSV] calculo_icc/df_{modelo[0].split('/')[1]}.csv", index=False)
            
            object = {
                "frases_ingles": [],
                "metrica": [],
                "modelo": [],
                "score": [],
            }
            df = montar_df(modelos_gerais, dataset, object, metricas, vet_newsmet)
            df.to_csv(f"dataset_{dataset}/[CSV] calculo_icc/df_geral.csv", index=False)

        for metrica in metricas:
            gerar_icc(dataset, "prompt1", metrica.split("/")[0])
            gerar_icc(dataset, "prompt2", metrica.split("/")[0])
            gerar_icc(dataset, "geral", metrica.split("/")[0])