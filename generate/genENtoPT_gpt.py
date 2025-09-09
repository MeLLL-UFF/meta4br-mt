import pandas as pd
import json
import openai
import os

df = pd.read_csv("comparacao_datasets/newsmet.csv")

api_key = os.environ.get("OPENAI_TOKEN")

if not api_key :
    print("Variável de ambiente OPENAI_TOKEN não encontrada.")

client = openai.OpenAI(
    api_key = api_key
)

anotacoes = []
for frase in df['Text']:

    prompt1 = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso"
    prompt2 = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso. A frase pode ou não conter metáfora"
    prompt3 = f"Você é um especialista em metáforas e tradução criativa. Traduza {frase} para o português, mantendo o sentido metafórico original. Responda apenas com a tradução."
    prompt4 = f"Você é um especialista em metáforas e tradução criativa. Somente traduza {frase} para o português, mantendo o sentido metafórico original. Por exemplo, 'kick the bucket' deve ser traduzido como 'bater as botas', e não como 'chutar o balde'. Responda apenas com a tradução."


    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt4},
            ],
            max_tokens=200
    )

    response_gpt = response.choices[0].message.content

    result = {
        "fraseEN": frase,
        "traducaoPT": response_gpt
    }
    print(result)
    anotacoes.append(result)

    # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
    with open('dataset_newsmet/gpt/prompt4/ENtoPT.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)


    
  

