import transformers
import json
from google import genai

client = genai.Client(api_key="AIzaSyAX8TX9GS9o7842SWgBPqG8tkQa6OOZjVM")

with open('dataset_manualdata/gemini/prompt2/frases_traduzidas.json', 'r', encoding='utf-8') as file:
    dados = json.load(file)

vetor = []
i = 1

for objeto in dados:
    original = objeto['ingles_original']
    traduzido = objeto['ingles_traduzido']

    tokens_original_qtd = client.models.count_tokens(
        model="gemini-2.0-flash-lite", 
        contents=original 
    )
    tokens_traduzido_qtd = client.models.count_tokens(
        model="gemini-2.0-flash-lite", 
        contents=traduzido 
    )

    objeto['num_tokens_original'] = tokens_original_qtd.total_tokens
    objeto['num_tokens_traduzido'] = tokens_traduzido_qtd.total_tokens

    vetor.append(objeto)
    print(f"Frase {i}")
    i += 1

with open('dataset_manualdata/gemini/prompt2/frases_traduzidas.json', 'w', encoding='utf-8') as file:
    json.dump(vetor, file, ensure_ascii=False, indent=4)

