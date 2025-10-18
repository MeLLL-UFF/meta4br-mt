import pandas as pd
import numpy as np
import json

#Um ranking sum geral por modelo/prompt
def calcular_vetor_soma_por_prompt(modelos, prompt_id, dataset):
    soma_total = 0
    vetor_soma_final = []
    for modelo in modelos:
        if modelo in ["gemmaX", "marian", "meta"]:
            matriz = pd.read_csv(f'dataset_{dataset}/{modelo}/matriz.csv')
        else:
            matriz = pd.read_csv(f'dataset_{dataset}/{modelo}/{prompt_id}/matriz.csv')

        frases_originais = matriz.iloc[:, 0].tolist()
        vetor_soma_final.append(matriz.iloc[:, -1].tolist())

    # temos uma matriz, onde cada linha tem os ranking para cada modelo
    # gemini 1 2 6 3 6 3 6 
    # gpt 2 6 7 4 7 4 
    
    # soma os valores de cada coluna
    soma_por_coluna = []
    for i in range(len(vetor_soma_final[0])):  # Para cada posição (coluna)
        soma_coluna = sum(linha[i] for linha in vetor_soma_final)
        soma_por_coluna.append(soma_coluna)  

    # Criar relação entre frase e ranking total
    resultado = []
    for i in range(len(soma_por_coluna)):
        resultado.append({
            "frase": frases_originais[i],
            "ranking_total": soma_por_coluna[i]
        })      

    df_resultado = pd.DataFrame(resultado).sort_values('ranking_total', ascending=True)

    return df_resultado

#Um ranking sum geral por modelo/prompt
def calcular_vetor_soma_geral(modelos, prompt_ids, dataset):
    soma_total = 0
    vetor_soma_final = []
    for modelo in modelos:
        for prompt in prompt_ids:

            if modelo in ["gemmaX", "marian", "meta"]:
                matriz = pd.read_csv(f'dataset_{dataset}/{modelo}/matriz.csv')
                if prompt == "prompt2": #rodar 1 vez só qm é de tradução pq n tem varios prompts
                    break
            else:
                matriz = pd.read_csv(f'dataset_{dataset}/{modelo}/{prompt}/matriz.csv')

            frases_originais = matriz.iloc[:, 0].tolist()
            vetor_soma_final.append(matriz.iloc[:, -1].tolist())

    # temos uma matriz, onde cada linha tem os ranking para cada modelo
    # gemini 1 2 6 3 6 3 6 
    # gpt 2 6 7 4 7 4 
    
    # soma os valores de cada coluna
    soma_por_coluna = []
    for i in range(len(vetor_soma_final[0])):  # Para cada posição (coluna)
        soma_coluna = sum(linha[i] for linha in vetor_soma_final)
        soma_por_coluna.append(soma_coluna)  

    # Criar relação entre frase e ranking total
    resultado = []
    for i in range(len(soma_por_coluna)):
        resultado.append({
            "frase": frases_originais[i],
            "ranking_total": soma_por_coluna[i]
        })      

    df_resultado = pd.DataFrame(resultado).sort_values('ranking_total', ascending=True)

    return df_resultado

def definir_quartis(df_soma_total):
    # Pegar a coluna 'ranking_total' e calcular quartis
    rankings = df_soma_total['ranking_total']
    
    q1 = rankings.quantile(0.25)
    q2 = rankings.quantile(0.50) 
    q3 = rankings.quantile(0.75)
    
    # Classificar as frases por quartil
    df_soma_total['quartil'] = pd.cut(rankings, 
                                    bins=[-np.inf, q1, q2, q3, np.inf], 
                                    labels=['Q1_melhor', 'Q2', 'Q3', 'Q4_pior'])
    
    return df_soma_total

def selecionar_100_frases(df_quartis, dataset, end_name):

    # Pegar 25 frases aleatórias de cada quartil
    # Por que random_state=42? Para garantir que a seleção seja sempre a mesma em execuções diferentes. Toda vez que você rodar o script, as mesmas 25 frases serão selecionadas. Outros pesquisadores podem reproduzir os resultados exatos. Útil para comparações e validação científica.
    q1_selecionadas = df_quartis[df_quartis['quartil'] == 'Q1_melhor'].sample(25, random_state=42)
    q2_selecionadas = df_quartis[df_quartis['quartil'] == 'Q2'].sample(25, random_state=42)
    q3_selecionadas = df_quartis[df_quartis['quartil'] == 'Q3'].sample(25, random_state=42)
    q4_selecionadas = df_quartis[df_quartis['quartil'] == 'Q4_pior'].sample(25, random_state=42)
    
    frases_finais = pd.concat([q1_selecionadas, q2_selecionadas, q3_selecionadas, q4_selecionadas], ignore_index=True)

    # Ordenar por ranking_total (do melhor para o pior)
    frases_finais = frases_finais.sort_values('ranking_total', ascending=True)
    
    # Resetar índice após ordenação
    frases_finais = frases_finais.reset_index(drop=True)

    frases_finais.to_csv(f'dataset_{dataset}/[CSV] selecao_frases_criticas/frases_quartis_{end_name}.csv', index=False)        
    
# Para usar:
if __name__ == "__main__":
    modelos = ["gemini", "gemma3", "gpt", "llama", "mistral", "qwen", "gemmaX", "meta", "marian"]
    prompt_ids = ["prompt1", "prompt2", "prompt3", "prompt4"] 

    for dataset in ["manual_data", "newsmet"]:
        for prompt in prompt_ids:
            df_soma_total_por_prompt = calcular_vetor_soma_por_prompt(modelos, prompt, dataset)
            df_quartis_por_prompt = definir_quartis(df_soma_total_por_prompt)
            selecionar_100_frases(df_quartis_por_prompt, dataset, prompt)

        df_soma_total_geral = calcular_vetor_soma_geral(modelos, prompt_ids, dataset)
        df_quartis_geral = definir_quartis(df_soma_total_geral)
        selecionar_100_frases(df_quartis_geral, dataset, "geral")
