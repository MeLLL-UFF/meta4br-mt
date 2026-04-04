import pandas as pd
import numpy as np
import json

datasets = ["manual_data", "newsmet"]
modelos_gerais = ["gemini", "gemma3", "gpt", "llama", "mistral", "qwen", "gemmaX", "meta", "marian"]
modelos_especificos = ["gemmaX", "meta", "marian"]

def pegar_melhores_piores(path, melhores_frases, piores_frases, primeiro_modelo):
    df = pd.read_csv(f"{path}matriz.csv")

    ultima_coluna = df.iloc[:, -1].values
    limite = len(ultima_coluna) // 2

    ranking_melhores = sorted([(i, int(val)) for i, val in enumerate(ultima_coluna) if i <= limite], key=lambda x: x[1])
    ranking_piores = sorted([(i, int(val)) for i, val in enumerate(ultima_coluna) if i > limite], key=lambda x: x[1], reverse=True)

    indices_melhores = set(i for i, _ in ranking_melhores)
    indices_piores = set(i for i, _ in ranking_piores)

    for i in range(len(df)):
        frase = df.iloc[i, 0]
        score = df.iloc[i, -1]
        if i in indices_melhores:
            if primeiro_modelo:
                melhores_frases.append({"frase": frase, "hits": 1, "score": score})
            else:
                for f in melhores_frases:
                    if f["frase"] == frase:
                        f["hits"] += 1
                        break
        elif i in indices_piores:
            if primeiro_modelo:
                piores_frases.append({"frase": frase, "hits": 1, "score": score})
            else:
                for f in piores_frases:
                    if f["frase"] == frase:
                        f["hits"] += 1
                        break

    return melhores_frases, piores_frases

def salvar_csv(frases_com_score, dataset, end_name, tipo):
    df = pd.DataFrame(frases_com_score, columns=["frase", "score"])
    df.to_csv(f"dataset_{dataset}/[CSV] selecao_frases_criticas/{tipo}_frases_{end_name}.csv", index=False, encoding='utf-8')

def escolher_frases(modelos, dataset, melhores_frases, piores_frases, end_name, prompt_id):
    for indice, modelo in enumerate(modelos):
        if prompt_id != "":
            path = f"dataset_{dataset}/{modelo}/{prompt_id}/"
        else:
            path = f"dataset_{dataset}/{modelo}/"
        primeiro = (indice == 0)
        melhores_frases, piores_frases = pegar_melhores_piores(path, melhores_frases, piores_frases, primeiro)
        print(f"Modelo {modelo}")

    n_modelos = len(modelos)
    melhores_unicas = {}
    piores_unicas = {}

    for f in melhores_frases:
        if f["hits"] == n_modelos:
            melhores_unicas[f["frase"]] = f["score"]
    for f in piores_frases:
        if f["hits"] == n_modelos:
            piores_unicas[f["frase"]] = f["score"]

    melhores_ordenadas = sorted(melhores_unicas.items(), key=lambda x: x[1])
    piores_ordenadas = sorted(piores_unicas.items(), key=lambda x: x[1], reverse=True)

    salvar_csv(melhores_ordenadas, dataset, end_name, "melhores")
    salvar_csv(piores_ordenadas, dataset, end_name, "piores")

    return melhores_ordenadas, piores_ordenadas

if __name__ == "__main__":
    for dataset in datasets:

        #modelos gerais com prompts diferentes
        for i in range(4): # 4 prompts atualmente
            melhores_frases = []
            piores_frases = []
            melhores_ordenadas, piores_ordenadas = escolher_frases(modelos_gerais, dataset, melhores_frases, piores_frases, f"prompt{i+1}_apenas", f"prompt{i+1}")

            todas_melhores += melhores_ordenadas
            todas_piores += piores_ordenadas

        #modelos específicos com mesmo prompt
        melhores_frases = []
        piores_frases = []
        melhores_ordenadas, piores_ordenadas = escolher_frases(modelos_especificos, dataset, melhores_frases, piores_frases, "prompt_unico", "")

        # Juntar os rankings e preservar o melhor score
        todas_melhores +=  melhores_ordenadas
        todas_piores +=  piores_ordenadas

        melhores_dict = {}
        for frase, score in todas_melhores:
            if frase not in melhores_dict or score < melhores_dict[frase]:
                melhores_dict[frase] = score
        piores_dict = {}
        for frase, score in todas_piores:
            if frase not in piores_dict or score > piores_dict[frase]:
                piores_dict[frase] = score

        melhores_final = sorted(melhores_dict.items(), key=lambda x: x[1])
        piores_final = sorted(piores_dict.items(), key=lambda x: x[1], reverse=True)

        pd.DataFrame([{"frase": frase} for frase, _ in melhores_final]).to_csv(f"dataset_{dataset}/[CSV] selecao_frases_criticas/melhores_frases_geral.csv", index=False, encoding='utf-8')

        pd.DataFrame([{"frase": frase} for frase, _ in piores_final]).to_csv(f"dataset_{dataset}/[CSV] selecao_frases_criticas/piores_frases_geral.csv", index=False, encoding='utf-8')

