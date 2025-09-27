import csv
import pandas as pd
import json

# Aqui eu adiciono cabeçalho e as colunas de frase_original, frase_traduzida_pt, frase_traduzida_en

for dataset in ["manual_data", "newsmet"]:
    for modelo in ["gemini", "gemma3", "gemmaX", "gpt", "llama", "marian", "meta", "mistral", "qwen"]:
        for prompt in ["prompt1", "prompt2", "prompt3", "prompt4"]:
            
            if modelo in ["gemmaX", "marian", "meta"]:
                path = f"dataset_{dataset}/{modelo}/"
                if prompt == "prompt2": #Rodar só uma vez
                    break 
            else:
                path = f"dataset_{dataset}/{modelo}/{prompt}/"
            print(path)

            header = ['ROUGE', 'BLEU', 'BERTSCORE', 'BLEURT', 'COMET22', 'KIWI-XL', 'XCOMET-XL']

            with open(path+'matriz.csv', "r", newline='', encoding="utf-8") as infile:
                linhas = list(csv.reader(infile))

            with open(path+'matriz.csv', 'w', newline='', encoding='utf-8') as f:
                escritor = csv.writer(f)
                escritor.writerow(header)  
                escritor.writerows(linhas) 

            with open(path+'frases_traduzidas_com_metricas.json', 'r', newline='') as file:
                arquivo_metricas = json.load(file)
            df = pd.read_csv(path+'matriz.csv', encoding='utf-8')

            num_orig_cols = len(df.columns)
            novas_colunas = pd.DataFrame('', index=df.index, columns=['ingles_original', 'portugues_traduzido', 'ingles_traduzido'])

            for i in range(len(df)):
                for col in novas_colunas.columns:
                    valor = arquivo_metricas[i][col]
                    novas_colunas.at[i, col] = valor

            df_final = pd.concat([novas_colunas, df], axis=1)

            df_final.to_csv(path+'matriz.csv', index=False)

            # Soma no final do CSV
            df = pd.read_csv(path+'matriz.csv', encoding='utf-8')
            df["Soma_ranking"] = 0

            for i in range(len(df)):
                df.at[i, "Soma_ranking"] = df.iloc[i, 3:].sum()
            
            df.to_csv(path+'matriz.csv', index=False)
