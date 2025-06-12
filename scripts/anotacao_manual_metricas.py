# Arquivo para calcular as métricas em cima da nossa anotação manual das frases

import pandas as pd
import json

modelos = ["gpt", "gemini",  "llama", "mistral", "qwen", "meta", "marian", "gemmaX"]

def calcular_metricas(df, modelo):
    print(df.columns.tolist())

    objeto = {
        "modelo": modelo,
        "total_frases": len(df),
        "contem_metafora": df["Contém metáfora?"].value_counts().to_dict(),
        "fluencia": df["Fluência"].value_counts().to_dict(),
        "inteligibilidade": df["Inteligibilidade"].value_counts().to_dict(),
        "compreensão": df["Compreensão"].value_counts().to_dict(),
        "equivalencia_total": df["Equivalência Total"].value_counts().to_dict(),
        "equivalencia_parcial": df["Equivalência Parcial"].value_counts().to_dict(),
        "nao_equivalencia": df["Não-equivalência"].value_counts().to_dict(),
        "traducao_incorreta": df["Tradução incorreta"].value_counts().to_dict(),
        "erro_fonte": df["Erro na fonte"].value_counts().to_dict(),
        "nao_traducao": df["Não tradução"].value_counts().to_dict(),
    }

    return objeto


def completar_json(objeto, arquivo_resultados):

    try:
        with open(arquivo_resultados, "r", encoding="utf-8") as f:
            dados = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        dados = []
            
    dados.append(objeto)

    with open(arquivo_resultados, "w") as f:
        json.dump(dados, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    for modelo in modelos:
        df = pd.read_csv(f"planilhas/anotacao_{modelo}.csv")
        objeto = calcular_metricas(df, modelo)
        completar_json(objeto, f"planilhas/resultados_anotacao_manual.json")
        print(f"Modelo {modelo} processado com sucesso, já foram calculadas as métricas.")

