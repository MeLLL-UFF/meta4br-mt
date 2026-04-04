import os, json, pandas as pd

# Montando matriz com os scores pra cada frase em cada modelo/prompt 
def montar_matriz_scores(modelos, prompts, dataset):
    matriz = []

    df = pd.read_csv(f"dataset_{dataset}/[CSV] selecao_frases_criticas/frases_quartis_geral.csv")

    for frase in df['frase']:
        linha = []
        for modelo in modelos:
            cont = 0
            
            for prompt in prompts:
            
                if modelo in ["gemmaX", "marian", "meta"]: 
                    cont +=1 
                    if cont > 1: break
                    prompt = ""
                else: prompt += "/"

                path = f"dataset_{dataset}/{modelo}/{prompt}frases_traduzidas_com_metricas.json"
                arq_metrica = pd.read_json(path)

                #Vou analisar em cima do XCOMET-XL que foi o com menores notas na análise do ICC

                linha.append({
                    'ingles_original': frase,
                    'portugues_traduzido': arq_metrica[arq_metrica['ingles_original'] == frase]['portugues_traduzido'].iloc[0],
                    'ingles_traduzido': arq_metrica[arq_metrica['ingles_original'] == frase]['ingles_traduzido'].iloc[0],
                    'modelo_prompt': f"{modelo}/{prompt.replace('/', '')}",
                    'XCOMET-XL': arq_metrica[arq_metrica['ingles_original'] == frase]['XCOMET-XL'].iloc[0]['scores']
                })


        # Pra depois pegar o menor e maior score por frase
        linha.sort(key=lambda x: x['XCOMET-XL'])
        matriz.append(linha)
    
    # Converter matriz em DataFrame
    rows = []
    for linha in matriz:
        for item in linha:
            rows.append(item)

    df_matriz = pd.DataFrame(rows)
    df_matriz.to_csv(f"dataset_{dataset}/[CSV] selecao_frases_criticas/matriz_scores_frases_quartis.csv", index=False)
    return

def montar_planilha_analise(dataset):

    df = pd.read_csv(f"dataset_{dataset}/[CSV] selecao_frases_criticas/matriz_scores_frases_quartis.csv")

    result = pd.DataFrame(columns=['ingles_original', 'portugues_traduzido', 'ingles_traduzido', 'modelo_prompt', 'XCOMET-XL'])

    for indice in range(0, len(df), 27):
        # Pegar o grupo de 27 linhas para a frase atual
        grupo_frase = df.iloc[indice:indice+27].copy()
        
        # Adicionar dados da primeira frase (menor score - índice 0, já que está ordenado crescentemente)
        primeira_frase = grupo_frase.iloc[0].copy()
        result = pd.concat([result, primeira_frase.to_frame().T], ignore_index=True)
        
        # Pegar o maior score (última linha do grupo, índice 26)
        maior_score = grupo_frase.iloc[26]['XCOMET-XL']
        
        # Pegar todos modelos e prompts que tiveram o maior score
        modelos_maior_score = ""
        for i in range(1, 26):  # Ignorar a primeira e última (0 e 26)
            if grupo_frase.iloc[i]['XCOMET-XL'] == maior_score:
                modelos_maior_score += f"{grupo_frase.iloc[i]['modelo_prompt']}; "

        # Modificar a última frase (maior score) para incluir todos os modelos que tiveram o mesmo score
        ultima_frase = grupo_frase.iloc[26].copy()
        modelo_original = str(ultima_frase['modelo_prompt']) if pd.notna(ultima_frase['modelo_prompt']) else ""
        ultima_frase['modelo_prompt'] = f"{modelo_original}; {modelos_maior_score}"
        result = pd.concat([result, ultima_frase.to_frame().T], ignore_index=True)

    result.to_excel(f"planilhas/anotacao_{dataset}_quartis.xlsx", index=False, engine='openpyxl')

    return

if __name__ == "__main__":
    dataset = ["manual_data", "newsmet"] 
    modelos = ["gemini", "gemma3", "gemmaX", "gpt", "llama", "marian", "meta", "mistral", "qwen"]
    prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]

    for dataset in dataset: 
        montar_matriz_scores(modelos, prompts, dataset)
        montar_planilha_analise(dataset)
        print(f"Finalizado o dataset {dataset}")
    
