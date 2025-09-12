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
        df = pd.read_csv("comparacao_datasets/newsmet.csv")
        term = "Text"
    case 2:
        dataset = "manual_data"
        df = pd.read_parquet("comparacao_datasets/manual_data.parquet")
        term = "Sentence"
    case _:
        print("Dataset inválido")
        exit()

model_id = "ModelSpace/GemmaX2-28-9B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

anotacoes = []
for frase in df[term]:

    prompt = f"Translate this from English to Portuguese:\English: {frase}\nPortuguese:"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = {
        "fraseEN": frase,
        "traducaoPT": response.split('Portuguese:')[2]
    }
    anotacoes.append(result)

    with open(f'dataset_{dataset}/gemmaX/ENtoPT.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)