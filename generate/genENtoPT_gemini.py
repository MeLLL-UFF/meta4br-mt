import pandas as pd
import json
import time
from google import genai
from google.genai import types

#pip install -U -q "google-genai"

df = pd.read_csv("comparacao_datasets/newsmet.csv", encoding='utf-8')

client = genai.Client(api_key="AIzaSyAX8TX9GS9o7842SWgBPqG8tkQa6OOZjVM")
delay = 15

anotacoes = []

for frase in df['Text'][2224:]:

    prompt1 = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso"
    prompt2 = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso. A frase pode ou não conter metáfora"


    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt1,
    )

    
    result = {
        "fraseEN": frase,
        "traducaoPT": response.text
    }

    anotacoes.append(result)

    # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
    with open('dataset_newsmet/gemini/prompt1/ENtoPT3.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)

    time.sleep(delay)

    
  

