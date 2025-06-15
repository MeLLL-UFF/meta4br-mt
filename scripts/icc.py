import pingouin as pg
import pandas as pd
import json
import os
import glob

# Um pra prompt1 e tradicionais e um pra prompt2 e tradicionais
# ICC1
# Fazer um geral pra todos os modelos, seja prompt1, prompt2 e tradicionais

def montar_df(modelos, dataset, object, metricas, frases, padrao):
    dados_modelos = {}
    
    # Lê todos os arquivos de uma vez e armazena em memória
    for modelo in modelos:
        caminho = f"dataset_{dataset}/{modelo}/frases_traduzidas_com_metricas.json"
        with open(caminho, "r") as f:
            dados_modelos[modelo] = json.load(f)
    
    for i, frase in enumerate(frases):
        if dataset == "newsmet":
            df_original = pd.read_csv(f"comparacao_datasets/{dataset}.csv")
            if ((df_original["Text"] == frase) & (df_original["predicted_label"] == padrao)).any():
                continue

        elif dataset == "manualdata":
            df_original = pd.read_parquet(f"comparacao_datasets/manual_data.parquet")
            if not ((df_original["Sentence"] == frase) & (df_original["Label"] == padrao)).any():
                continue

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
        
def gerar_icc(dataset, complemento, metrica, metaphor_or_not):
    path = f"dataset_{dataset}/[CSV]calculo_icc/df_{complemento}_{metaphor_or_not}.csv"
    if not os.path.exists(path):
        print(f"Arquivo {path} não encontrado. Pulando ICC para {complemento} - {metrica}.")
    df = pd.read_csv(path)

    df_metrica_atual = df[df["metrica"] == metrica]

    icc = pg.intraclass_corr(data=df_metrica_atual, targets='frases_ingles', raters='modelo', ratings='score').round(3)
    icc.to_csv(f"dataset_{dataset}/[CSV]calculo_icc/icc_{complemento}/icc_{metrica}_{complemento}_{metaphor_or_not}.csv", index=False)


def resumir_iccs(complemento, dataset, metaphor_or_not):
    pasta_icc = f"dataset_{dataset}/[CSV]calculo_icc/icc_{complemento}/{metaphor_or_not}"
    arquivos = [f for f in os.listdir(pasta_icc) if f.endswith(".csv")]

    resultados = []

    for nome_arquivo in arquivos:
        caminho_arquivo = os.path.join(pasta_icc, nome_arquivo)

        try:
            parte_nome = nome_arquivo.split("_")[1]
        except IndexError:
            print(f"❗ Nome inesperado: {nome_arquivo}")
            continue

        df_icc = pd.read_csv(caminho_arquivo)
        if "Type" not in df_icc.columns:
            print(f"❗ Coluna 'Type' não encontrada em {nome_arquivo}")
            continue

        linha_icc3k = df_icc[df_icc["Type"].str.strip().str.upper() == "ICC3K"]
        if not linha_icc3k.empty:
            valor_icc = linha_icc3k["ICC"].values[0]
            resultados.append({
                "arquivo": nome_arquivo,
                "metrica": parte_nome,
                "icc3k": valor_icc
            })
        else:
            print(f"⚠️ ICC3k não encontrado em {nome_arquivo}")

    if resultados:
        df_resultados = pd.DataFrame(resultados)
        caminho_saida = f"dataset_{dataset}/[CSV]calculo_icc/icc3k_{complemento}_{metaphor_or_not}_resumo.csv"
        df_resultados.to_csv(caminho_saida, index=False)
        print(f"📁 Resumo salvo em: {caminho_saida}")
    else:
        print("⚠️ Nenhum resultado coletado.")


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

        # df_com_metafora = pd.DataFrame()
        # df_sem_metafora = pd.DataFrame()
        # if dataset == "manualdata":
        #     for modelo in modelos:
        #         object = {
        #             "frases_ingles": [],
        #             "metrica": [],
        #             "modelo": [],
        #             "score": [],
        #         }
        #         df_com_metafora = montar_df(modelo, dataset, object, metricas, vet_manualdata, 0)
        #         df_com_metafora.to_csv(f"dataset_{dataset}/[CSV]calculo_icc/df_{modelo[0].split("/")[1]}_com_metafora.csv", index=False)
        #         df_sem_metafora = montar_df(modelo, dataset, object, metricas, vet_manualdata, 1)
        #         df_sem_metafora.to_csv(f"dataset_{dataset}/[CSV]calculo_icc/df_{modelo[0].split("/")[1]}_sem_metafora.csv", index=False)
                
        #     object = {
        #         "frases_ingles": [],
        #         "metrica": [],
        #         "modelo": [],
        #         "score": [],
        #     }

        #     df_com_metafora = montar_df(modelos_gerais, dataset, object, metricas, vet_manualdata, 0)
        #     df_com_metafora.to_csv(f"dataset_{dataset}/[CSV]calculo_icc/df_geral_com_metafora.csv", index=False)
        #     df_sem_metafora = montar_df(modelos_gerais, dataset, object, metricas, vet_manualdata, 1)
        #     df_sem_metafora.to_csv(f"dataset_{dataset}/[CSV]calculo_icc/df_geral_sem_metafora.csv", index=False)

        # elif dataset == "newsmet":
        #     for modelo in modelos:
        #         object = {
        #             "frases_ingles": [],
        #             "metrica": [],
        #             "modelo": [],
        #             "score": [],
        #         }
        #         df_com_metafora = montar_df(modelo, dataset, object, metricas, vet_newsmet, "literal")
        #         df_com_metafora.to_csv(f"dataset_{dataset}/[CSV]calculo_icc/df_{modelo[0].split('/')[1]}_com_metafora.csv", index=False)
        #         df_sem_metafora = montar_df(modelo, dataset, object, metricas, vet_newsmet, "metaphorical")
        #         df_sem_metafora.to_csv(f"dataset_{dataset}/[CSV]calculo_icc/df_{modelo[0].split('/')[1]}_sem_metafora.csv", index=False)
            
        #     object = {
        #         "frases_ingles": [],
        #         "metrica": [],
        #         "modelo": [],
        #         "score": [],
        #     }
        #     df_com_metafora = montar_df(modelos_gerais, dataset, object, metricas, vet_newsmet, "literal")
        #     df_com_metafora.to_csv(f"dataset_{dataset}/[CSV]calculo_icc/df_geral_com_metafora.csv", index=False)
        #     df_sem_metafora = montar_df(modelos_gerais, dataset, object, metricas, vet_newsmet, "metaphorical")
        #     df_sem_metafora.to_csv(f"dataset_{dataset}/[CSV]calculo_icc/df_geral_sem_metafora.csv", index=False)

        # for metrica in metricas:
        #     gerar_icc(dataset, "prompt1", metrica.split("/")[0], "com_metafora")
        #     gerar_icc(dataset, "prompt2", metrica.split("/")[0], "com_metafora")
        #     gerar_icc(dataset, "geral", metrica.split("/")[0], "com_metafora")
        #     gerar_icc(dataset, "prompt1", metrica.split("/")[0], "sem_metafora")
        #     gerar_icc(dataset, "prompt2", metrica.split("/")[0], "sem_metafora")
        #     gerar_icc(dataset, "geral", metrica.split("/")[0], "sem_metafora")

        
        # Resumir os ICCs
        resumir_iccs("prompt1", dataset, "com_metafora")
        resumir_iccs("prompt2", dataset, "com_metafora")
        resumir_iccs("geral", dataset, "com_metafora")
        resumir_iccs("prompt1", dataset, "sem_metafora")
        resumir_iccs("prompt2", dataset, "sem_metafora")
        resumir_iccs("geral", dataset, "sem_metafora")
