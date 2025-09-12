from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json

################################################################################################
#                                        AJUSTES INICIAIS

### Decidir qual dataset irá usar
dataset_id = 1 # 1 - newsmet | 2 - manual_data 

################################################################################################

match dataset_id:
    case 1:
        dataset = "newsmet"
    case 2:
        dataset = "manual_data"
    case _:
        print("Dataset inválido")
        exit()

with open(f'dataset_{dataset}/gemmaX/ENtoPT.json', 'r', encoding='utf-8') as f:
    dados = json.load(f)

model_id = "ModelSpace/GemmaX2-28-9B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

anotacoes = []
i = 1

for frase in dados:
    i += 1
    print(f"frase {i}\n")

    prompt = f"Translate this from Portuguese to English:\Portuguese: {frase['traducaoPT']}\English:"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=512)

    result = {
        "frasePT": frase['traducaoPT'],
        "traducaoEN": tokenizer.decode(outputs[0], skip_special_tokens=True).split('English:')[2]
    }
    anotacoes.append(result)

    with open(f'dataset_{dataset}/gemmaX/PTtoEN.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)