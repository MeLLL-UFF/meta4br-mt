import pingouin as pg
import pandas as pd

# Um pra prompt1 e tradicionais e um pra prompt2 e tradicionais
# ICC1
# Fazer um geral pra todos os modelos, seja prompt1, prompt2 e tradicionais

datasets = ["manualdata", "newsmet"]
modelos_prompt1 = ["gemini/prompt1/", "gemma3/prompt1/", "gemmaX", "gpt/prompt1/", "llama/prompt1/", "marian", "meta", "mistral/prompt1/", "qwen/prompt1/"]
modelos_prompt2 = ["gemini/prompt2/", "gemma3/prompt2/" "gemmaX", "gpt/prompt2/", "llama/prompt2/" "marian", "meta", "mistral/prompt2/","qwen/prompt2/"]

def calcular_icc(prompt, modelos, dataset, df, object, tam, metricas):
    if modelo in ["marian", "meta", "gemmaX"]:
        arquivo = f"dataset_{dataset}/{modelo}/frases_traduzidas_com_metricas.json"
    else:
        arquivo = f"dataset_{dataset}/{modelo}/{prompt}/frases_traduzidas_com_metricas.json"
    dados = pd.read_json(arquivo, encoding='utf-8')
    for i,frase in enumerate(dados[1]):
        object["frases_ingles"].append(frase["ingles_original"])*tam
        for metrica in metricas:
            object["metricas"].append(metrica)
        for modelo in modelos:
            object["rouge"].append(dados[i]["ROUGE"]["rougeL"])
            object["bleu"].append(dados[i]["BLEU"]["bleu"])
            object["bertscore"].append(dados[i]["BERTSCORE"]["f1"])
            object["bleurt"].append(dados[i]["BLEURT"]["scores"])
            object["comet22"].append(dados[i]["COMET22"]["scores"])
            object["kiwi_xl"].append(dados[i]["KIWI-XL"]["scores"])
            object["xcomet_xl"].append(dados[i]["XCOMET-XL"]["scores"])
        

if __name__ == "__main__":
    # data = pg.read_dataset('icc')
    # icc = pg.intraclass_corr(data=data, targets='Wine', raters='Judge', ratings='Scores').round(3)

    metricas = ["ROUGE", "BLEU", "BERTSCORE", "BLEURT", "COMET22", "KIWI-XL", "XCOMET-XL"]
    for dataset in datasets:
        object = {
            "frases_ingles": [],
            "metricas": [],
            "rouge": [],
            "bleu": [],
            "bertscore": [],
            "bleurt": [],
            "comet22": [],
            "kiwi_xl": [],
            "xcomet_xl": []
        }

        df = pd.DataFrame()

        df = calcular_icc(modelos_prompt1, dataset, df, object, len(modelos_prompt1), metricas)
        df.to_csv(f"dataset_{dataset}/icc.csv", index=False)
        print(object)

        # df = calcular_icc(modelos_prompt2, dataset, df, object, len(modelos_prompt2), metricas)
        # df.to_csv(f"dataset_{dataset}/icc.csv", index=False)

