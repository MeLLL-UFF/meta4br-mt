# -*- coding: utf-8 -*-
import pandas as pd
import os

# --- Configure suas variáveis aqui ---
modelos = ["gemini", "gemma3", "gpt", "llama", "mistral", "qwen", "gemmaX", "meta", "marian"]
prompt_ids = ["prompt1", "prompt2", "prompt3", "prompt4"]
datasets = ["manual_data", "newsmet"]
# ------------------------------------

def verificar_tamanho_matrizes(datasets, modelos, prompt_ids):
    """
    Verifica e imprime o shape (linhas, colunas) de cada arquivo matriz.csv.
    """
    for dataset in datasets:
        print(f"\n=========================================")
        print(f" VERIFICANDO DATASET: {dataset}")
        print(f"=========================================")
        
        for modelo in modelos:
            print(f"\n--- Modelo: {modelo} ---")
            
            # Modelos com estrutura de pasta simples (tradução)
            if modelo in ["gemmaX", "marian", "meta"]:
                caminho_arquivo = f'dataset_{dataset}/{modelo}/matriz.csv'
                imprimir_shape(caminho_arquivo)
            
            # Modelos com subpastas para cada prompt
            else:
                for prompt in prompt_ids:
                    caminho_arquivo = f'dataset_{dataset}/{modelo}/{prompt}/matriz.csv'
                    imprimir_shape(caminho_arquivo)

def imprimir_shape(caminho_arquivo):
    """
    Lê um arquivo CSV e imprime seu shape. Lida com arquivos não encontrados.
    """
    try:
        # Verifica se o arquivo existe antes de tentar ler
        if os.path.exists(caminho_arquivo):
            matriz = pd.read_csv(caminho_arquivo)
            # A propriedade .shape retorna uma tupla (linhas, colunas)
            print(f"Arquivo: {caminho_arquivo:<55} | Shape: {matriz.shape}")
        else:
            print(f"Arquivo: {caminho_arquivo:<55} | !!! ARQUIVO NÃO ENCONTRADO !!!")
            
    except Exception as e:
        print(f"Arquivo: {caminho_arquivo:<55} | *** ERRO AO LER O ARQUIVO: {e} ***")


if __name__ == "__main__":
    verificar_tamanho_matrizes(datasets, modelos, prompt_ids)