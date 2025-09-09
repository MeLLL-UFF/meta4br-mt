# import pandas as pd
# import json
# import time
# from google import genai
# from google.genai import types

# #pip install -U -q "google-genai"

# client = genai.Client(
#     vertexai=True,
#     project="metaphor-459717",
#     location="us-central1",
# )

# delay = 15
# anotacoes = []

# with open('dataset_newsmet/gemini/prompt2/ENtoPT.json', 'r', encoding='utf-8') as f:
#     vetor = json.load(f)

# for objeto in vetor:

#     prompt1 = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso"
#     prompt2 = f"Traduzir a frase '{objeto['traducaoPT']}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso. A frase pode ou não conter metáfora"

#     response = client.models.generate_content(
#         model="gemini-2.0-flash-lite",
#         contents=prompt2,
#     )
    
#     result = {
#         "frasePT": objeto['traducaoPT'],
#         "traducaoEN": response.text
#     }
#     anotacoes.append(result)

#     # Isso aqui acaba reescrevendo o json mil vezes, mas é bom pq se der problema na máquina, não perco todas as frases, consigo continuar de onde parei
#     with open('dataset_newsmet/gemini/prompt2/PTtoEN.json', 'w', encoding='utf-8') as f:
#         json.dump(anotacoes, f, ensure_ascii=False, indent=5)

#     time.sleep(delay)

    
  

"""

python3 generate/genPTtoEN_gemini.py --input_dataset dataset_newsmet/gemini/prompt4/ENtoPT.json  --output_base dataset_newsmet/gemini --start_index 0 --batch_size 5 --max_workers 5 --sleep 2

"""

import os, json, time, argparse, concurrent.futures
from pathlib import Path
import pandas as pd
from google import genai
from google.genai import types
import os

client = genai.Client(
    vertexai=True,
    project= "metaphor-459717",
    location="us-east1",
)

MODEL = "gemini-2.0-flash-lite"

def build_config():
    return types.GenerateContentConfig(
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="OFF"),
        ],
    )

def _fetch_translation(args_tuple):
    frase, prompt_text, sleep_time = args_tuple
    time.sleep(sleep_time)
    try:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])]
        response = client.models.generate_content(model=MODEL, contents=contents, config=build_config())
        translated = response.text.strip()

    except Exception as e:
        translated = f"Error: {e}"
    return dict(frasePT=frase, traducaoEN=translated)

def batched_translation(frases, prompt_template, batch_size, sleep_time, max_workers, out_path, resultados_existentes):
    resultados = resultados_existentes.copy()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, len(frases), batch_size):
            end = min(start + batch_size, len(frases))
            batch = frases[start:end]
            args_iter = [(f, prompt_template.format(f), sleep_time) for f in batch]

            novos_resultados = list(executor.map(_fetch_translation, args_iter))
            resultados.extend(novos_resultados)

            # Salvar após cada batch
            with open("dataset_newsmet/gemini/prompt4/PTtoEN.json", 'w', encoding='utf-8') as f:
                json.dump(resultados, f, ensure_ascii=False, indent=5)

            print(f"Batch {start}-{end} salvo em {out_path}")

    return resultados


def main(args):
    with open(args.input_dataset, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    frases_pt = []
    for objeto in dados[args.start_index:]:
        frases_pt.append(objeto["traducaoPT"])
    print(len(frases_pt))


    prompt1 = "Traduzir a frase '{}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso."
    prompt2 = "Traduzir a frase '{}' do português para o inglês. Apenas escreva a frase traduzida, nada além disso. A frase pode ou não conter metáfora."
    prompt3 = "Você é um especialista em metáforas e tradução criativa. Traduza '{}' para o inglês, mantendo o sentido metafórico original. Responda apenas com a tradução."
    prompt4 = "Você é um especialista em metáforas e tradução criativa. Somente traduza '{}' para o inglês, mantendo o sentido metafórico original. Por exemplo, 'kick the bucket' deve ser traduzido como 'bater as botas', e não como 'chutar o balde'. Responda apenas com a tradução."

    for prompt_id, prompt_template in enumerate([prompt4], start=4): #mudar aqui enumerate([prompt1, prompt2], start=1
        out_path = Path(args.output_base) / f"prompt{str(prompt_id)}" / f"PTtoEN.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        resultados = []

        if out_path.exists():
            print(f"Retomando a partir de {out_path}")
            with open(out_path, 'r', encoding='utf-8') as f:
                resultados = json.load(f)

        frases_processadas = set(r["frasePT"] for r in resultados if isinstance(r, dict) and "frasePT" in r)
        frases_faltantes = [f for f in frases_pt if f not in frases_processadas]

        novos_resultados = batched_translation(
            frases=frases_faltantes,
            prompt_template=prompt_template,
            batch_size=args.batch_size,
            sleep_time=args.sleep,
            max_workers=args.max_workers,
            out_path=out_path,
            resultados_existentes=resultados
        )

        resultados.extend(novos_resultados)

        print(f"\nTraduções com Prompt {prompt_id} salvas em {out_path} ({len(resultados)} frases)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", required=True)
    parser.add_argument("--output_base", required=True,
                        help="Diretório base onde serão salvos os arquivos ENtoPT1.json, ENtoPT2.json etc.")
    parser.add_argument("--column", default="Sentence")
    parser.add_argument("--start_index", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=2)
    parser.add_argument("--max_workers", type=int, default=5)

    main(parser.parse_args())