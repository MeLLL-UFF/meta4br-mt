#pip install groq

import json
from groq import Groq
import pandas as pd
import os

api_key = os.environ.get("GROQ_TOKEN")

if api_key :
    print("Token lido:", api_key)
else:
    print("Variável de ambiente GROQ_TOKEN não encontrada.")

client = Groq(
    api_key = api_key
)

df = pd.read_parquet("comparacao_datasets/manual_data.parquet", engine='pyarrow') 

with open('groq/ENtoPT.json', 'r', encoding='utf-8') as f:
    dados = json.load(f)

anotacoes = []

for objeto in dados:

    prompt1 = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso"

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt1,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    result = {
        "frasePT": objeto['traducaoPT'],
        "traducaoEN": response.choices[0].message.content
    }

    anotacoes.append(result)

    # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
    with open('groq/PTtoEN.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)


