import json
import tiktoken

with open('dataset_manualdata/gpt/prompt2/frases_traduzidas.json', 'r', encoding='utf-8') as file:
    dados = json.load(file)

vetor = []
i = 1

for objeto in dados:
    original = objeto['ingles_original']
    traduzido = objeto['ingles_traduzido']

    tokens_original_num = encoding.encode(original)
    tokens_traduzido_num = encoding.encode(traduzido)

    objeto['tokens_original'] = [encoding.decode([t]) for t in tokens_original_num]
    objeto['num_tokens_original'] = len(tokens_original_num)
    objeto['tokens_traduzido'] = [encoding.decode([t]) for t in tokens_traduzido_num]
    objeto['num_tokens_traduzido'] = len(tokens_traduzido_num)

    vetor.append(objeto)
    print(f"Frase {i}")
    i += 1

with open('dataset_manualdata/gpt/prompt2/frases_traduzidas.json', 'w', encoding='utf-8') as file:
    json.dump(vetor, file, ensure_ascii=False, indent=4)

