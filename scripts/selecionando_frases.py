import pandas as pd
import numpy as np
import json

datasets = ["manualdata", "newsmet"]
modelos1 = ["gemini/prompt1", "gemma3/prompt1", "gpt/prompt1", "llama/prompt1", "mistral/prompt1", "qwen/prompt1", "gemmaX", "meta", "marian"]
modelos2 = ["gemini/prompt2", "gemma3/prompt2", "gpt/prompt2", "llama/prompt2", "mistral/prompt2", "qwen/prompt2", "gemmaX", "meta", "marian"]
modelos3 = ["gemmaX", "meta", "marian"]


def pegar_melhores_piores(path, melhores_frases, piores_frases, primeiro_modelo):
    df = pd.read_csv(f"{path}matriz.csv")

    # Esse cálculo de limite tá bom ???
    ultima_coluna = df.iloc[:, -1].values
    limite = len(ultima_coluna) // 2

    # Com o limite, as frases não se cruzam!!
    ranking_melhores = sorted([int(elem) for i, elem in enumerate(ultima_coluna) if i <= limite])
    ranking_piores = sorted([int(elem) for i, elem in enumerate(ultima_coluna) if i > limite], reverse=True)

    
    for i in range(len(df)):
        if(primeiro_modelo):
            if(df.iloc[i, -1] in ranking_melhores):
                melhores_frases.append({"frase": df.iloc[i, 0], "hits" : 1})
            elif(df.iloc[i, -1] in ranking_piores):
                piores_frases.append({"frase": df.iloc[i, 0], "hits" : 1})
        else:
            if(df.iloc[i, -1] in ranking_melhores):
                for frase in melhores_frases:
                    if(frase["frase"] == df.iloc[i, 0]):
                        frase["hits"] += 1
            elif(df.iloc[i, -1] in ranking_piores):
                for frase in piores_frases:
                    if(frase["frase"] == df.iloc[i, 0]):
                        frase["hits"] += 1

    return melhores_frases, piores_frases

def salvar_csv(melhores_frases, piores_frases, dataset, prompt_id):
    melhores_df = pd.DataFrame(melhores_frases, columns=["frase"])
    piores_df = pd.DataFrame(piores_frases, columns=["frase"])

    melhores_df.to_csv(f"dataset_{dataset}/melhores_frases_{prompt_id}.csv", index=False, encoding='utf-8')
    piores_df.to_csv(f"dataset_{dataset}/piores_frases_{prompt_id}.csv", index=False, encoding='utf-8')

def escolher_frases(modelos, dataset, melhores_frases, piores_frases, prompt_id):
    for indice, modelo in enumerate(modelos):
        path = f"dataset_{dataset}/{modelo}/"
        if(indice == 0):
            melhores_frases, piores_frases = pegar_melhores_piores(path, melhores_frases, piores_frases, True)
        else:
            melhores_frases, piores_frases = pegar_melhores_piores(path, melhores_frases, piores_frases, False)
        print(f"Modelo {modelo}")
    
    print(f"Melhores {melhores_frases}")
    print(f"Piores {piores_frases}")

    # Deixando só as frases que aparecem em todos os modelos
    melhores_frases = [obj["frase"] for obj in melhores_frases if obj["hits"] == len(modelos) or obj["hits"] == 2*len(modelos)]
    piores_frases = [obj["frase"] for obj in piores_frases if obj["hits"] == len(modelos) or obj["hits"] == 2 *len(modelos)]

    salvar_csv(melhores_frases, piores_frases, dataset, prompt_id)

if __name__ == "__main__":
    for dataset in datasets:
        melhores_frases = []
        piores_frases = []
        # A comparação para achar as frases finais que deverão ser anotadas deve analisar prompt1 e 2 juntos ou separados? tipo, pego a interseçao entre prompt1 e prompt2 mesmo né?
        escolher_frases(modelos1, dataset, melhores_frases, piores_frases, "prompt1_apenas")
        melhores_frases = []
        piores_frases = []
        escolher_frases(modelos2, dataset, melhores_frases, piores_frases, "prompt2_apenas")
        melhores_frases = []
        piores_frases = []
        escolher_frases(modelos3, dataset, melhores_frases, piores_frases, "prompt_unico")

