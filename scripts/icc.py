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
        if len(dados_modelos[modelo]) == 0:
            print(f"Modelo {modelo} do dataset {dataset} não possui dados.")
            break
    
    for i, frase in enumerate(frases):
        # print(f"Frase {i+1} de {len(frases)}")
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

def check_missing_models(df, expected_models, dataset_name):
    # Extrair os nomes base dos modelos esperados (removendo '/prompt1/' etc)
    expected_models_base = [m.split('/')[0] for m in expected_models]
    expected_models_base = list(set(expected_models_base))  # Remover duplicados
    
    # Modelos presentes nos dados
    present_models = df['modelo'].unique()
    
    # Modelos faltantes
    missing_models = [m for m in expected_models_base if m not in present_models]
    
    print(f"\nDataset: {dataset_name}")
    print("Modelos esperados:", expected_models_base)
    print("Modelos presentes:", present_models)
    print("Modelos faltantes:", missing_models)
    
    return missing_models

def check_models_per_sentence(df, sentence_index):
        frase = df['frases_ingles'].unique()[sentence_index]
        models_for_sentence = df[df['frases_ingles'] == frase]['modelo'].unique()
        print(f"\nFrase: {frase}")
        print("Modelos presentes:", models_for_sentence)
        print("Modelos faltantes:", [m for m in all_models if m not in models_for_sentence])
        

if __name__ == "__main__":
    datasets = ["manualdata", "newsmet"]
    modelos_prompt1 = ["gemini/prompt1/", "gemma3/prompt1/", "gemmaX", "gpt/prompt1/", "llama/prompt1/", "marian", "meta", "mistral/prompt1/", "qwen/prompt1/"]
    modelos_prompt2 = ["gemini/prompt2/", "gemma3/prompt2/" "gemmaX", "gpt/prompt2/", "llama/prompt2/" "marian", "meta", "mistral/prompt2/","qwen/prompt2/"]
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
            df.to_csv(f"dataset_{dataset}/icc_prompt1.csv", index=False)
        elif dataset == "newsmet":
            df = montar_df(modelos_prompt1, dataset, df, object, metricas, vet_newsmet)
            df.to_csv(f"dataset_{dataset}/icc_prompt2.csv", index=False)
    df_manualdata = pd.read_csv("dataset_manualdata/icc.csv")
    print(df_manualdata.columns)
    print(df_manualdata.head())

    df_newsmet = pd.read_csv("dataset_newsmet/icc.csv")
    print(df_manualdata.columns)
    print(df_manualdata.head())

    # Verifica se todas as frases têm o mesmo número de modelos
    contagem = df_manualdata.groupby("frases_ingles")["modelo"].nunique()
    print(contagem.value_counts())

    # Verifica se todas as frases têm o mesmo número de modelos
    contagem = df_newsmet.groupby("frases_ingles")["modelo"].nunique()
    print(contagem.value_counts())

    # Lista de modelos esperados (ajuste conforme necessário)
    all_models = modelos_prompt1 + modelos_prompt2
    all_models = list(set([m.split('/')[0] for m in all_models]))  # Obter nomes únicos

    # Verificar para cada dataset
    missing_manual = check_missing_models(df_manualdata, modelos_prompt1, "manualdata (prompt1)")
    missing_newsmet = check_missing_models(df_newsmet, modelos_prompt1, "newsmet (prompt1)")

    print(missing_manual)
    print(missing_newsmet)
        # Verificar quantas frases cada modelo possui
    print("\nContagem de frases por modelo (manualdata):")
    print(df_manualdata.groupby('modelo')['frases_ingles'].nunique())

    print("\nContagem de frases por modelo (newsmet):")
    print(df_newsmet.groupby('modelo')['frases_ingles'].nunique())


    icc_manualdata = pg.intraclass_corr(data=df_manualdata, targets='frases_ingles', raters='modelo', ratings='score').round(3)
    df_manualdata.to_csv("dataset_manualdata/icc_final.csv", index=False)
    print(df_manualdata.head())

    icc_newsmet = pg.intraclass_corr(data=df_newsmet, targets='frases_ingles', raters='modelo', ratings='score').round(3)
    df_newsmet.to_csv("dataset_newsmet/icc_final.csv", index=False)
    print(df_newsmet.head())

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