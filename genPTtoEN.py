import pandas as pd
import json
import openai


client = openai.OpenAI(
    api_key="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"
)

anotacoes = []

output_path = '--output_path'
arquivo_entrada = output_path + '/ENtoPT.json'
arquivo_saida = output_path + '/PTtoEN.json'

with open(arquivo_entrada, 'r', encoding='utf-8') as f:
    vetor = json.load(f)

for objeto in vetor:

#     prompt = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês"

#     response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "user", "content": prompt},
#             ],
#             max_tokens=2000
#     )

#     response_gpt = response.choices[0].message.content

#     result = {
#         "frasePT": objeto['traducaoPT'],
#         "traducaoEN": response_gpt
#     }

#     anotacoes.append(result)

# # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
# with open(arquivo_saida, 'w', encoding='utf-8') as f:
#     json.dump(anotacoes, f, ensure_ascii=False, indent=5)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run text generation with Hugging Face pipeline.")
#     parser.add_argument('--model_id', type=str, required=True, help="Hugging Face Model ID.")
#     parser.add_argument('--hf_token', type=str, required=True, help="Hugging Face API token.")
#     parser.add_argument('--output_path', type=str, required=True, help="Path to save the generated outputs.")

#     args = parser.parse_args()

#     main(args.model_id, args.hf_token, args.output_path)



