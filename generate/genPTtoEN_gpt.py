import pandas as pd
import json
import openai
import os

api_key = os.environ.get("OPENAI_TOKEN")

if api_key :
    print("Token lido:", api_key)
else:
    print("Variável de ambiente OPENAI_TOKEN não encontrada.")

client = openai.OpenAI(
    api_key = api_key
)

anotacoes = []

with open('dataset_newsmet/gpt/prompt2/ENtoPT.json', 'r', encoding='utf-8') as f:
    vetor = json.load(f)

for objeto in vetor:

    prompt1 = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso"
    prompt2 = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso. A frase pode ou não conter metáfora"

    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt2},
            ],
            max_tokens=2000
    )

    response_gpt = response.choices[0].message.content
    
    result = {
        "frasePT": objeto['traducaoPT'],
        "traducaoEN": response_gpt
    }
    anotacoes.append(result)

    # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
    with open('dataset_newsmet/gpt/prompt2/PTtoEN.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)


    
  

