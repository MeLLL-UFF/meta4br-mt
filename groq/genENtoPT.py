#pip install groq

import json
from groq import Groq
import pandas as pd

df = pd.read_parquet("comparacao_datasets/manual_data.parquet", engine='pyarrow') 

client = Groq(
    api_key="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"
)

anotacoes = []

for frase in df['Sentence']:

    prompt = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso"

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    result = {
        "fraseEN": frase,
        "traducaoPT": response.choices[0].message.content
    }

    anotacoes.append(result)

    # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
    with open('groq/ENtoPT.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)


