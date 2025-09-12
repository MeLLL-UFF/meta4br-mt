# pip3 install evaluate 
# pip3 install rouge-score 
# pip3 install absl-py 
# pip3 install nltk 
# pip3 install bert_score 
# pip3 install git+https://github.com/google-research/bleurt.git
# pip3 install unbabel-comet

# pip3 install evaluate rouge-score absl-py nltk bert_score git+https://github.com/google-research/bleurt.git unbabel-comet

import evaluate
import json
import torch
import gc

def calcular_metricas(dataset_name, model_name, prompt_id):

    if model_name in ["gemmaX", "marian", "meta"]: # Os modelos de tradução, que não possuem prompt
        prompt_id = ""
    else:
        prompt_id = f"/{prompt_id}"

    torch.cuda.empty_cache()
    gc.collect()

    rouge = evaluate.load('rouge')
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    bleurt = evaluate.load('bleurt', 'bleurt-large-512')

    with open(f'dataset_{dataset_name}/{model_name}{prompt_id}/frases_traduzidas.json', 'r', encoding='utf-8') as file:
        dados = json.load(file)

    vetor = []
    i = 1

    for objeto in dados:

        prediction = objeto["ingles_traduzido"]
        reference = objeto["ingles_original"]
        source = objeto["portugues_traduzido"]

        result_rouge = rouge.compute(predictions=[prediction], references=[reference])
        result_bleu = bleu.compute(predictions=[prediction], references=[reference])
        result_bertscore = bertscore.compute(predictions=[prediction], references=[reference], model_type="distilbert-base-uncased")
        result_bleurt = bleurt.compute(predictions=[prediction], references=[reference])

        result = {
            "ingles_original": reference,
            "portugues_traduzido": source,
            "ingles_traduzido": prediction,
            "ROUGE": {
                "rouge1" : result_rouge["rouge1"],
                "rouge2" : result_rouge["rouge2"],
                "rougeL" : result_rouge["rougeL"],
                "rougeLsum" : result_rouge["rougeLsum"],
            },
            "BLEU": {
                "bleu" : result_bleu["bleu"],
                "precisions" : result_bleu["precisions"],
                "brevity_penalty" : result_bleu["brevity_penalty"],
                "length_ratio" : result_bleu["length_ratio"],
                "translation_lenght" : result_bleu["translation_length"],
                "reference_lenght" : result_bleu["reference_length"]
            },
            "BERTSCORE": {
                "precision" : result_bertscore["precision"][0],
                "recall" : result_bertscore["recall"][0],
                "f1" : result_bertscore["f1"][0],
                "hashcode" : result_bertscore["hashcode"]
            },
            "BLEURT": {
                "scores" : result_bleurt["scores"][0],
            }
        }

        vetor.append(result)

        print(f"frase {i}\n")
        i += 1

        with open(f'dataset_{dataset_name}/{model_name}{prompt_id}/frases_traduzidas_com_metricas.json', 'w', encoding='utf-8') as file:
            json.dump(vetor, file, ensure_ascii=False, indent=4)

        print(f'Arquivo salvo em dataset_{dataset_name}/{model_name}{prompt_id}/frases_traduzidas_com_metricas.json')

#Se não quer juntar de uma vez pra todos os datasets, todos os modelos e todos os prompts (foi fazendo pro partes), deve alterar os FORs abaixo)
if __name__ == "__main__":
    for dataset in ["newsmet", "manual_data"]:
        for model in ["gemini", "gemma3", "gpt", "llama", "mistral", "qwen"]:
            for prompt in ["prompt1", "prompt2", "prompt3", "prompt4"]:
                if model in ["gemma3", "marian", "meta"]: # Os modelos de tradução, que não possuem prompt
                    if prompt == "prompt2": # Só roda 1 vez para os modelos de tradução
                        break
                calcular_metricas(dataset, model, prompt)
