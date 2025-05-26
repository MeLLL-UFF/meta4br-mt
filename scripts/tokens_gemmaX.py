from transformers import AutoTokenizer
import transformers
import json
from google import genai

with open('dataset_newsmet/gemmaX/frases_traduzidas.json', 'r', encoding='utf-8') as file:
    dados = json.load(file)

model_name = "ModelSpace/GemmaX2-28-9B-v0.1"
vetor = []
i = 1

for objeto in dados:
    original = objeto['ingles_original']
    traduzido = objeto['ingles_traduzido']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens_original = tokenizer.tokenize(original)
    tokens_traduzido = tokenizer.tokenize(traduzido)

    objeto['tokens_original'] = tokens_original
    objeto['num_tokens_original'] = len(tokens_original)
    objeto['tokens_traduzido'] = tokens_traduzido
    objeto['num_tokens_traduzido'] = len(tokens_traduzido)

    vetor.append(objeto)
    print(f"Frase {i}")
    i += 1

    with open('dataset_newsmet/gemmaX/frases_traduzidas.json', 'w', encoding='utf-8') as file:
        json.dump(vetor, file, ensure_ascii=False, indent=4)

