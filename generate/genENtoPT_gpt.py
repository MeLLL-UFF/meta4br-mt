import pandas as pd
import json
import openai
import os

#pip install --upgrade openai 

################################################################################################
#                                      AJUSTES INICIAIS
### Decidir qual dataset irá usar
dataset_id = 1 # 1 - newsmet | 2 - manual_data 

### Nome do prompt e nome da pasta que serão salvas as saídas das LLMs
prompt_id = "prompt3" 

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

api_key = os.environ.get("OPENAI_TOKEN")

if not api_key :
    print("Variável de ambiente OPENAI_TOKEN não encontrada.")

client = openai.OpenAI(
    api_key = api_key
)

anotacoes = []
for frase in df[term]:

    prompt1 = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso"
    prompt2 = f"Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso. A frase pode ou não conter metáfora"
    prompt3 = f"Você é um especialista em metáforas e tradução criativa. Traduza {frase} para o português, mantendo o sentido metafórico original. Responda apenas com a tradução."
    prompt4 = f"Você é um especialista em metáforas e tradução criativa. Somente traduza {frase} para o português, mantendo o sentido metafórico original. Por exemplo, 'kick the bucket' deve ser traduzido como 'bater as botas', e não como 'chutar o balde'. Responda apenas com a tradução."

    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt_id},
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

    with open(f'dataset_{dataset_id}/gpt/{prompt_id}/ENtoPT.json', 'w', encoding='utf-8') as f:
        json.dump(anotacoes, f, ensure_ascii=False, indent=5)


    
  

