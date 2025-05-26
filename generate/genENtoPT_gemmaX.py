from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json

df = pd.read_parquet("comparacao_datasets/manual_data.parquet")

model_id = "ModelSpace/GemmaX2-28-9B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

anotacoes = []
for frase in df['Sentence'][463:]:

    prompt = f"Translate this from English to Portuguese:\English: {frase}\nPortuguese:"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = {
        "fraseEN": frase,
        "traducaoPT": response.split('Portuguese:')[2]
    }
    anotacoes.append(result)

    # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
    with open('dataset_manualdata/gemmaX/ENtoPT1.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)