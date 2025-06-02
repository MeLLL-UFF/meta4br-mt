import csv
import pandas as pd
import json

datasets = ["newsmet", "manualdata"]
modelos = ["gemini/prompt1", "gemini/prompt2", "gemma3/prompt1", "gemma3/prompt2", "gemmaX", "gpt/prompt1", "gpt/prompt2", "llama/prompt1", "llama/prompt2", "marian", "meta", "mistral/prompt1", "mistral/prompt2", "qwen/prompt1", "qwen/prompt2"]

for dataset in datasets:
    for modelo in modelos:
        arquivo_matriz = f"dataset_{dataset}/{modelo}/matriz.csv"
        print(arquivo_matriz)

        header = ['ROUGE', 'BLEU', 'BERTSCORE', 'BLEURT', 'COMET22', 'KIWI-XL', 'XCOMET-XL']

        with open(arquivo_matriz, "r", newline='', encoding="utf-8") as infile:
            linhas = list(csv.reader(infile))

        with open(arquivo_matriz, 'w', newline='', encoding='utf-8') as f:
            escritor = csv.writer(f)
            escritor.writerow(header)  
            escritor.writerows(linhas) 

        path_metricas = f"dataset_{dataset}/{modelo}/frases_traduzidas_com_metricas.json"
        with open(path_metricas, 'r', newline='') as file:
            arquivo_metricas = json.load(file)
        df = pd.read_csv(arquivo_matriz)

        num_orig_cols = len(df.columns)
        novas_colunas = pd.DataFrame('', index=df.index, columns=['ingles_original', 'portugues_traduzido', 'ingles_traduzido'])

        for i in range(len(df)):
            for col in novas_colunas.columns:
                valor = arquivo_metricas[i][col]
                novas_colunas.at[i, col] = valor

        df_final = pd.concat([novas_colunas, df], axis=1)

        df_final.to_csv(arquivo_matriz, index=False)

        # Soma no final do CSV
        df = pd.read_csv(arquivo_matriz, encoding='utf-8')
        df["Soma_ranking"] = 0

        for i in range(len(df)):
            df.at[i, "Soma_ranking"] = df.iloc[i, 3:].sum()
            
        df.to_csv(arquivo_matriz, index=False)
