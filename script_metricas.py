# pip install evaluate 
# pip install rouge-score 
# pip install absl-py 
# pip install nltk 
# pip install bert_score 
# pip install git+https://github.com/google-research/bleurt.git
# pip install unbabel-comet

import evaluate
import json
import torch
import gc

torch.cuda.empty_cache()
gc.collect()

rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
bleurt = evaluate.load('bleurt', 'bleurt-large-512')

with open('qwen/frases_traduzidas.json', 'r', encoding='utf-8') as file:
    dados = json.load(file)

vetor = []

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

    with open('qwen/frases_traduzidas_com_metricas.json', 'w', encoding='utf-8') as file:
        json.dump(vetor, file, ensure_ascii=False, indent=4)

