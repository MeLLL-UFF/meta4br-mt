import pandas as pd
import json
from google import genai

#pip install -U -q "google-genai"

df = pd.read_parquet("comparacao_datasets/common1.parquet", engine='pyarrow') 

client = genai.Client(api_key="AIzaSyAX8TX9GS9o7842SWgBPqG8tkQa6OOZjVM")

with open('gemini/ENtoPT.json', 'r', encoding='utf-8') as f:
    anotacoes = json.load(f)

# anotacoes = []

for frase in df['text'][10:20]:

    # prompt = f"Traduzir a frase '{frase}' do inglês para o português"
    prompt = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso"


    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt,
    )

    
    result = {
        "fraseEN": frase,
        "traducaoPT": response.text
    }

    anotacoes.append(result)

    # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
    with open('gemini/ENtoPT.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)


    
  

