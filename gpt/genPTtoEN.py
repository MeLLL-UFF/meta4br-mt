import pandas as pd
import json
import openai

client = openai.OpenAI(
    api_key="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"
)

anotacoes = []

with open('gpt/ENtoPT_manualdata.json', 'r', encoding='utf-8') as f:
    vetor = json.load(f)

for objeto in vetor:

    prompt = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso"

    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000
    )

    response_gpt = response.choices[0].message.content
    
    result = {
        "frasePT": objeto['traducaoPT'],
        "traducaoEN": response_gpt
    }
    anotacoes.append(result)

    # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
    with open('gpt/PTtoEN_manualdata.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)


    
  

