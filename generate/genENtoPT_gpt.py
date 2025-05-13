import pandas as pd
import json
import openai


df = pd.read_csv("comparacao_datasets/newsmet.csv", encoding='utf-8')

client = openai.OpenAI(
    api_key="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"
)

anotacoes = []
for frase in df['Text']:

    prompt1 = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso"
    prompt2 = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso. A frase pode ou não conter metáfora"


    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt2},
            ],
            max_tokens=2000
    )

    response_gpt = response.choices[0].message.content
    
    result = {
        "fraseEN": frase,
        "traducaoPT": response_gpt
    }
    anotacoes.append(result)

    # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
    with open('dataset_newsmet/gpt/prompt2/ENtoPT.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)


    
  

