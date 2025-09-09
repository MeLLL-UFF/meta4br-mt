import pandas as pd
import json
import time
from google import genai
from google.genai import types
import os

#pip3 install -U -q "google-genai"

df = pd.read_csv("comparacao_datasets/newsmet.csv")

client = genai.Client(
    vertexai=True,
    project= "metaphor-459717",
    location="us-east1",
)

prompt_id = "prompt3"

# delay = 10 
anotacoes = []

for frase in df['Text']:

    match prompt_id:
        case "prompt1":
            prompt = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso"
        case "prompt2":
            prompt = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso. A frase pode ou não conter metáfora"
        case "prompt3":
            prompt = f"Você é um especialista em metáforas e tradução criativa. Traduza {frase} para o português, mantendo o sentido metafórico original. Responda apenas com a tradução."
        case "prompt4":
            prompt = f"Você é um especialista em metáforas e tradução criativa. Somente traduza {frase} para o português, mantendo o sentido metafórico original. Por exemplo, 'kick the bucket' deve ser traduzido como 'bater as botas', e não como 'chutar o balde'. Responda apenas com a tradução."


    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt,
    )

    
    result = {
        "fraseEN": frase,
        "traducaoPT": response.text
    }

    print(result)
    anotacoes.append(result)

    # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
    with open(f'dataset_newsmet/gemini/{prompt_id}/ENtoPT.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)

    # time.sleep(delay)

    
  

