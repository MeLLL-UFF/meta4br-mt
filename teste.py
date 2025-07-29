import pandas as pd
import os

def process_phrases():
    """
    Processa o arquivo 'Piores frases - manual_data.csv' para encontrar o melhor modelo de tradução para cada frase
    com base na pontuação ranksum de vários arquivos 'matriz.csv' específicos do modelo.

    A função lê um arquivo CSV principal de frases, itera através de cada um e pesquisa em
    vários arquivos 'matriz.csv' para encontrar o modelo com o menor ranksum para essa frase.
    Em seguida, preenche o dataframe principal com o nome do melhor modelo e suas traduções.

    Retorna:
        O caminho para o arquivo CSV recém-criado com os dados processados.
    """
    try:
        # Carrega a planilha principal
        main_df = pd.read_csv('Análises do Back Translation.csv')
    except FileNotFoundError:
        return "Arquivo 'Análises do Back Translation.csv' não encontrado. Verifique se o arquivo está no mesmo diretório do script."

    # Define os caminhos para os arquivos matriz.csv de cada modelo
    model_paths = [
        # alternar entre manualdata e newsmet
        'dataset_newsmet/gemma3/prompt2/matriz.csv',
        'dataset_newsmet/mistral/prompt2/matriz.csv',
        'dataset_newsmet/gemini/prompt2/matriz.csv',
        'dataset_newsmet/gpt/prompt2/matriz.csv',
        'dataset_newsmet/qwen/prompt2/matriz.csv',
        'dataset_newsmet/llama/prompt2/matriz.csv',
        'dataset_newsmet/gemmaX/matriz.csv',
        'dataset_newsmet/marian/matriz.csv',
        'dataset_newsmet/meta/matriz.csv'
        
    ]

    # Cria novas colunas no dataframe principal se elas ainda não existirem
    main_df['modelo'] = ''
    main_df['portugues_traduzido'] = ''
    main_df['ingles_traduzido'] = ''

    # Itera sobre cada frase no dataframe principal
    for index, row in main_df.iterrows():
        phrase = row['ingles_original']
        lowest_ranksum = float('inf')
        best_model_data = {}

        # Procura a frase em cada matriz.csv do modelo
        for path in model_paths:
            try:
                model_df = pd.read_csv(path)
                # Assumindo que a primeira coluna é a frase original em inglês e a última é o ranksum
                phrase_row = model_df[model_df.iloc[:, 0] == phrase]

                if not phrase_row.empty:
                    ranksum = phrase_row.iloc[0, -1]
                    if ranksum < lowest_ranksum:
                        lowest_ranksum = ranksum
                        
                        # Extrai o nome do modelo do caminho
                        path_parts = path.split('/')
                        model_name = path_parts[-2] if 'prompt' not in path_parts[-2] else f"{path_parts[-3]}"
                        
                        best_model_data = {
                            'modelo': model_name,
                            'portugues_traduzido': phrase_row.iloc[0, 1], # Assumindo que a tradução para o português é a 2ª coluna
                            'ingles_traduzido': phrase_row.iloc[0, 2]    # Assumindo que a retrotradução é a 3ª coluna
                        }
            except FileNotFoundError:
                # Ignora silenciosamente se um arquivo matriz.csv não for encontrado
                continue

        # Atualiza o dataframe principal com os dados do melhor modelo encontrado
        if best_model_data:
            main_df.loc[index, 'modelo'] = best_model_data['modelo']
            main_df.loc[index, 'portugues_traduzido'] = best_model_data['portugues_traduzido']
            main_df.loc[index, 'ingles_traduzido'] = best_model_data['ingles_traduzido']

    # Salva o dataframe atualizado em um novo arquivo CSV
    output_filename = 'frases_processadas.csv'
    main_df.to_csv(output_filename, index=False)

    return output_filename

# Executa a função de processamento e imprime o resultado
output_file = process_phrases()
print(f"Processamento concluído! O resultado foi salvo no arquivo: {output_file}")