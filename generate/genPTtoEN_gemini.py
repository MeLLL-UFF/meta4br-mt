import pandas as pd
import json
import time
from google import genai
from google.genai import types

#pip install -U -q "google-genai"

client = genai.Client(
    vertexai=True,
    project="metaphor-459717",
    location="us-central1",
)

delay = 15
anotacoes = []

with open('dataset_newsmet/gemini/prompt1/ENtoPT.json', 'r', encoding='utf-8') as f:
    vetor = json.load(f)

for objeto in vetor[662:]:

    prompt1 = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso"
    prompt2 = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso. A frase pode ou não conter metáfora"

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt1,
    )
    
    result = {
        "frasePT": objeto['traducaoPT'],
        "traducaoEN": response.text
    }
    anotacoes.append(result)

    # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
    with open('dataset_newsmet/gemini/prompt1/PTtoEN1.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)

    time.sleep(delay)

    
  

