import pandas as pd
import json
import openai
import os

################################################################################################
#                                      AJUSTES INICIAIS
#pip install --upgrade openai 

### Decidir qual dataset irá usar
dataset_id = 1 # 1 - newsmet | 2 - manual_data 

### Nome do prompt e nome da pasta que serão salvas as saídas das LLMs
prompt_id = "prompt3" 

# Só ajustando as variáveis pro código ficar mais automatizado, com menos alterações
match dataset_id:
    case 1:
        dataset = "newsmet"
    case 2:
        dataset = "manual_data"
    case _:
        print("Dataset inválido")
        exit()

################################################################################################
api_key = os.environ.get("OPENAI_TOKEN")

if not api_key :
    print("Variável de ambiente OPENAI_TOKEN não encontrada.")

client = openai.OpenAI(
    api_key = api_key
)

anotacoes = []

with open(f'dataset_{dataset}/gpt/{prompt_id}/ENtoPT.json', 'r', encoding='utf-8') as f:
    vetor = json.load(f)

for objeto in vetor:

    prompt1 = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso"
    prompt2 = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso. A frase pode ou não conter metáfora"
    prompt3 = f"Você é um especialista em metáforas e tradução criativa. Traduza '{objeto['traducaoPT']}' para o inglês, mantendo o sentido metafórico original. Responda apenas com a tradução."
    prompt4 = f"Você é um especialista em metáforas e tradução criativa. Somente traduza '{objeto['traducaoPT']}' para o inglês, mantendo o sentido metafórico original. Por exemplo, 'kick the bucket' deve ser traduzido como 'bater as botas', e não como 'chutar o balde'. Responda apenas com a tradução."

    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt_id},
            ],
            max_tokens=200
    )

    response_gpt = response.choices[0].message.content
    
    result = {
        "frasePT": objeto['traducaoPT'],
        "traducaoEN": response_gpt
    }
    anotacoes.append(result)
    print(result)

    # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
    with open(f'dataset_{dataset}/gpt/{prompt_id}/PTtoEN.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)


    
  

