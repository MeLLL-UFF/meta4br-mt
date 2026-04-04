import json
import pandas as pd
import os

datasets = ["manual_data", "newsmet"]
modelos = ["gemini", "gemma3", "gemmaX", "gpt", "llama", "marian", "meta", "mistral", "qwen"]
prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]

def buscar_label(frase, arq_dataset, dataset):

    if dataset == "manual_data":
        resultado = arq_dataset[arq_dataset['Sentence'] == frase]
        if not resultado.empty:
            return int(resultado['Label'].values[0])
        else:
            return None
    else:
        resultado = arq_dataset[arq_dataset['Text'] == frase]
        if not resultado.empty:
            return 1 if resultado['sentence_label'].values[0] == "metaphorical" else 0
        else:
            return None

def preencher_matriz(arquivo):
    with open(arquivo, 'r') as file:
        dados = json.load(file)

    print(f"Criando para o arquivo {arquivo}...")

    # Ordenar frases pelas métricas (mantém lógica original)
    metricas = [
        {"metrica": "ROUGE", "vetor": sorted(set([obj["ROUGE"]["rougeL"] for obj in dados]), reverse=True)},
        {"metrica": "BLEU", "vetor": sorted(set([obj["BLEU"]["bleu"] for obj in dados]), reverse=True)},
        {"metrica": "BERTSCORE", "vetor": sorted(set([obj["BERTSCORE"]["f1"] for obj in dados]), reverse=True)},
        {"metrica": "BLEURT", "vetor": sorted(set([obj["BLEURT"]["scores"] for obj in dados]), reverse=True)},
        {"metrica": "COMET22", "vetor": sorted(set([obj["COMET22"]["scores"] for obj in dados]), reverse=True)},
        {"metrica": "KIWI-XL", "vetor": sorted(set([obj["KIWI-XL"]["scores"] for obj in dados]), reverse=True)},
        {"metrica": "XCOMET-XL", "vetor": sorted(set([obj["XCOMET-XL"]["scores"] for obj in dados]), reverse=True)}
    ]
    auxiliar = ["rougeL", "bleu", "f1", "scores", "scores", "scores", "scores"]

    # Descobrir dataset
    if "manual_data" in arquivo:
        dataset = "manual_data"
        arq_dataset = pd.read_parquet("comparacao_datasets/manual_data.parquet")
    elif "newsmet" in arquivo:
        dataset = "newsmet"
        arq_dataset = pd.read_csv("comparacao_datasets/newsmet.csv")
    else:
        dataset = None
        arq_dataset = None

    linhas = []
    for objeto_i in dados:
        
        ingles_original = objeto_i.get('ingles_original')
        portugues_traduzido = objeto_i.get('portugues_traduzido')
        ingles_traduzido = objeto_i.get('ingles_traduzido')
        label = buscar_label(ingles_original, arq_dataset, dataset) if dataset and ingles_original is not None else None
        metricas_vals = []

        for metrica_j in range(7):
            indice = metricas[metrica_j]["vetor"].index(objeto_i[metricas[metrica_j]["metrica"]][auxiliar[metrica_j]])
            metricas_vals.append(indice + 1)
        linha = [ingles_original, portugues_traduzido, ingles_traduzido, label] + metricas_vals
        linhas.append(linha)

    colunas_metricas = [m["metrica"] for m in metricas]
    colunas = ['ingles_original', 'portugues_traduzido', 'ingles_traduzido', 'label'] + colunas_metricas
    matriz = pd.DataFrame(linhas, columns=colunas)

    # Calcular Soma_ranking (soma das métricas)
    matriz["Soma_ranking"] = matriz[colunas_metricas].sum(axis=1)

    ordem = [
        'ingles_original',
        'portugues_traduzido',
        'ingles_traduzido',
        'label',
        'ROUGE',
        'BLEU',
        'BERTSCORE',
        'BLEURT',
        'COMET22',
        'KIWI-XL',
        'XCOMET-XL',
        'Soma_ranking'
    ]
    matriz = matriz[ordem]

    pasta = os.path.dirname(arquivo)
    path = os.path.join(pasta, "matriz.csv")
    matriz.to_csv(path, index=False)

def preencher_objeto(prompt, modelo, dataset):
    matriz = []

    if modelo in ["marian", "meta", "gemmaX"]:
        arquivo = f"dataset_{dataset}/{modelo}/frases_traduzidas_com_metricas.json"
    else:
        arquivo = f"dataset_{dataset}/{modelo}/{prompt}/frases_traduzidas_com_metricas.json"
    
    preencher_matriz(arquivo)

if __name__ == "__main__":
    for dataset in datasets:
        for modelo in modelos:
            for prompt in prompts:
                preencher_objeto(prompt, modelo, dataset)


