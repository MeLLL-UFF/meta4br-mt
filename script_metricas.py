# pip install rouge-score

import evaluate
import json

rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore", lang="en")
bleurt = evaluate.load("bleurt", module_type="metric")

with open('BackTranslation/gpt/frases_traduzidas.json', 'r', encoding='utf-8') as file:
    dados = json.load(file)

for objeto in dados[:2]:
    prediction = objeto["ingles_traduzido"]
    reference = objeto["ingles_original"]
    result_rouge = rouge.compute(predictions=[prediction], references=[reference])
    result_bleu = bleu.compute(predictions=[prediction], references=[reference])
    result_bertscore = bertscore.compute(predictions=[prediction], references=[reference])
    result_bleurt = bleurt.compute(predictions=[prediction], references=[reference])

    print(objeto)

    objeto["Rouge"] = {
        objeto["rouge1"] : result_rouge["rouge1"],
        objeto["rouge2"] : result_rouge["rouge2"],
        objeto["rougeL"] : result_rouge["rougeL"],
        objeto["rougeLsum"] : result_rouge["rougeLsum"],
    }

    objeto["Bleu"] = {
        objeto["bleu"] : result_bleu["bleu"],
        objeto["brevity_penality"] : result_bleu["brevity_penality"],
        objeto["lenght_ratio"] : result_bleu["lenght_ratio"],
    }

    objeto["BertScore"] = {
        objeto["precision"] : result_bertscore["precision"],
        objeto["f1"] : result_bertscore["f1"],
    }

    objeto["Bleurt"] = {
        objeto["scores"] : result_bleurt["scores"],
    }

    # Só pra ir vendo que está rodando algo, pq só escreve no arquivo no finald as 600

with open('BackTranslation/gpt/frases_traduzidas.json', 'w', encoding='utf-8') as file:
    json.dump(dados, file, ensure_ascii=False, indent=4)

