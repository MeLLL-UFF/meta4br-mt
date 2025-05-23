from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json

with open('dataset_newsmet/gemmaX/ENtoPT.json', 'r', encoding='utf-8') as f:
    dados = json.load(f)

model_id = "ModelSpace/GemmaX2-28-9B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

anotacoes = []
i = 0
for frase in dados['traducaoPT']:
    i += 1
    print(f"frase {i}\n")

    prompt = f"Translate this from Portuguese to English:\Portuguese: {frase}\English:"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=512)

    result = {
        "frasePT": frase,
        "traducaoEN": tokenizer.decode(outputs[0], skip_special_tokens=True)
    }
    anotacoes.append(result)

    with open('dataset_newsmet/gemmaX/PTtoEN.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)