import transformers
import json

with open('dataset_manualdata/llama/frases_traduzidas.json', 'r', encoding='utf-8') as file:
    dados = json.load(file)

tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)

vetor = []
i = 1

for objeto in dados:
    frase = objeto['ingles_original']

    tokens = tokenizer.tokenize(frase)

    objeto['tokens'] = tokens
    objeto['num_tokens'] = len(tokens)

    vetor.append(objeto)
    print(f"Frase {i}: {frase}")
    i += 1

with open('dataset_manualdata/llama/frases_traduzidas.json', 'w', encoding='utf-8') as file:
    json.dump(vetor, file, ensure_ascii=False, indent=4)

