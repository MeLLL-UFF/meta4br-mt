import json
import pandas as pd
import csv

respostas = {
    "manualdata_prompt1": [],
    "manualdata_prompt2": [],
    "newsmet_prompt1": [],
    "newsmet_prompt2": []
}

datasets = ["manualdata", "newsmet"]
modelos = ["gemini", "gemma3", "gemmaX", "gpt", "llama", "marian", "meta", "mistral", "qwen"]


def preencher_matriz(arquivo):
    matriz = []

    with open(arquivo, 'r') as file:
        dados = json.load(file)

    # Ordenando em ordem descrecente as frases com base no valor das métricas
    #Fiz um dicionário para não perder as metricas e retirei valores repetidos pra não interferir no ranking
    metricas = [
        {"metrica": "ROUGE", "vetor": sorted(set([obj["ROUGE"]["rougeL"] for obj in dados]),reverse=True)},
        {"metrica": "BLEU", "vetor": sorted(set([obj["BLEU"]["bleu"] for obj in dados]),reverse=True)},
        {"metrica": "BERTSCORE", "vetor": sorted(set([obj["BERTSCORE"]["f1"] for obj in dados]),reverse=True)},
        {"metrica": "BLEURT", "vetor": sorted(set([obj["BLEURT"]["scores"] for obj in dados]),reverse=True)},
        {"metrica": "COMET22", "vetor": sorted(set([obj["COMET22"]["scores"] for obj in dados]),reverse=True)},
        {"metrica": "KIWI-XL", "vetor": sorted(set([obj["KIWI-XL"]["scores"] for obj in dados]),reverse=True)},
        {"metrica": "XCOMET-XL", "vetor": sorted(set([obj["XCOMET-XL"]["scores"] for obj in dados]),reverse=True)}
    ]

    auxiliar = ["rougeL", "bleu", "f1", "scores", "scores", "scores", "scores"]    
    matriz = []
    for objeto_i in dados:
        frase_x = []
        for metrica_j in range(len(objeto_i.keys())-3):  # -3 para ignorar as chaves que não são métricas
            indice = metricas[metrica_j]["vetor"].index(objeto_i[metricas[metrica_j]["metrica"]][auxiliar[metrica_j]]) #pra achar o valor do dicionário objeto_i
            frase_x.append(indice + 1)  # +1 para começar a contagem do índice em 1
        matriz.append(frase_x)
    
    escrever_matriz_em_csv(matriz, arquivo)
              

def preencher_objeto(vet_prompt, modelo, dataset):
    matriz = []

    if modelo in ["marian", "meta", "gemmaX"]:
        arquivo = f"dataset_{dataset}/{modelo}/frases_traduzidas_com_metricas.json"
    else:
         arquivo = f"dataset_{dataset}/{modelo}/{vet_prompt[-7:]}/frases_traduzidas_com_metricas.json"

    object = {
        "modelo" : modelo,
        "dataset": dataset,
        "matriz" : preencher_matriz(arquivo)
    }

    return object


def escrever_matriz_em_csv(matriz, arquivo):
    path = f"{arquivo[0:arquivo.rfind('/frases_traduzidas_com_metricas.json')]}/matriz.csv"
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(matriz)


def print_matriz(matriz):
    # Calcula a largura máxima de cada coluna
    colunas = list(zip(*matriz))
    larguras = [max(len(str(item)) for item in coluna) for coluna in colunas]

    # Função para imprimir uma linha separadora
    def print_linha():
        print("+" + "+".join("-" * (largura + 2) for largura in larguras) + "+")

    # Imprime a matriz formatada
    print_linha()
    for linha in matriz:
        print("| " + " | ".join(f"{str(item):<{largura}}" for item, largura in zip(linha, larguras)) + " |")
        print_linha() 


for dataset in datasets:
        for modelo in modelos:
            # respostas[f"{dataset}_prompt1"].append(preencher_objeto(f"{dataset}_prompt1", modelo, dataset))
            respostas[f"{dataset}_prompt2"].append(preencher_objeto(f"{dataset}_prompt2", modelo, dataset))

