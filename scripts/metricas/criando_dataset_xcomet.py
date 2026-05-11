import os
import pandas as pd
import numpy as np
import json
from parascore import ParaScorer
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys

# Modelos e prompts já selecionados, os melhores
MODELOS = ["gpt", "gemini", "gemma3", "gemmaX"]
PROMPTS_POR_MODELO = {
    "gpt": ["prompt2", "prompt4"],
    "gemini": ["prompt2", "prompt4"],
    "gemma3": ["prompt2", "prompt4"],
    "gemmaX": [None],  
}
DATASETS = ["newsmet", "manual_data"]
QUARTIS_LABELS = ["Q4", "Q3", "Q2", "Q1"]  
N_QUARTIS_SUPERIORES = 2  


def ordenar_por_indice(df):
    if df is None or df.empty or "indice" not in df.columns:
        return df

    colunas_ordenacao = ["indice"]

    if "modelo" in df.columns:
        df = df.assign(_modelo_ordem=df["modelo"].map(get_modelo_idx))
        colunas_ordenacao.append("_modelo_ordem")

    if "prompt" in df.columns and "modelo" in df.columns:
        df = df.assign(
            _prompt_ordem=[get_prompt_idx(modelo, prompt) for modelo, prompt in zip(df["modelo"], df["prompt"])]
        )
        colunas_ordenacao.append("_prompt_ordem")

    df = df.sort_values(colunas_ordenacao).reset_index(drop=True)
    colunas_auxiliares = [coluna for coluna in ["_modelo_ordem", "_prompt_ordem"] if coluna in df.columns]
    if colunas_auxiliares:
        df = df.drop(columns=colunas_auxiliares)
    return df

def get_modelo_idx(modelo):
    ordem_modelos = list(PROMPTS_POR_MODELO.keys())
    try:
        return ordem_modelos.index(modelo)
    except ValueError:
        return len(ordem_modelos)  # sempre maior que qualquer índice válido

def get_prompt_idx(modelo, prompt):
    ordem_prompts = PROMPTS_POR_MODELO.get(modelo, [])
    if prompt == "sem_prompt":
        # Se "sem_prompt" está explicitamente na ordem, retorna o índice, senão, vai para o final
        if None in ordem_prompts:
            return ordem_prompts.index(None)
        return len(ordem_prompts)
    try:
        return ordem_prompts.index(prompt)
    except ValueError:
        return len(ordem_prompts)

def processar_dataset(dataset_nome):
    print(f"\nProcessando dataset: {dataset_nome}")
    print("Selecionando melhores frases por combinação de COMET22, XCOMET-XL e KIWI-XL")
    df = selecionar_melhor_combinado(dataset_nome)
    if df is None or df.empty:
        print(f"  [AVISO] Nenhuma frase encontrada para {dataset_nome}")
        return

    saida_dir = os.path.join(f"dataset_{dataset_nome}", "[CSV] melhor_xcomet")
    os.makedirs(saida_dir, exist_ok=True)

    saida = os.path.join(saida_dir, "melhor_combinado.csv")
    df_ordenado = ordenar_por_indice(df.copy())
    df_ordenado.to_csv(saida, index=False)

    gerar_datasets_finais_por_label(df, dataset_nome, saida_dir)
    calcular_parascore_dataset_final(dataset_nome)
    plotar_distribuicao_parascore(dataset_nome)
    plotar_distribuicao_score_combinado(dataset_nome)

def selecionar_melhor_combinado(dataset_nome):
  
    frases_dict = {}
    for modelo in MODELOS:
        prompts = PROMPTS_POR_MODELO[modelo]
        for prompt in prompts:
            if prompt:
                metricas_path = f"dataset_{dataset_nome}/{modelo}/{prompt}/frases_traduzidas_com_metricas.json"
            else:
                metricas_path = f"dataset_{dataset_nome}/{modelo}/frases_traduzidas_com_metricas.json"
            if not os.path.exists(metricas_path):
                continue
            with open(metricas_path, "r") as f:
                dados = json.load(f)
            for idx, obj in enumerate(dados):
                ingles = obj["ingles_original"]
                label = obj["label"]
                chave_frase = (label, ingles)
                portugues = obj["portugues_traduzido"]
                
                score_combinado = (obj["COMET22"]["scores"] + obj["XCOMET-XL"]["scores"] + obj["KIWI-XL"]["scores"]) / 3.0
                if chave_frase not in frases_dict or score_combinado > frases_dict[chave_frase]["score_combinado"]:
                    frases_dict[chave_frase] = {
                        "indice": idx,
                        "ingles_original": ingles,
                        "portugues_traduzido": portugues,
                        "label": label,
                        "dataset": dataset_nome,
                        "modelo": modelo,
                        "prompt": prompt if prompt else "sem_prompt",
                        "score_combinado": score_combinado,
                        "xcomet_xl": obj["XCOMET-XL"]["scores"],
                        "comet22": obj["COMET22"]["scores"],
                        "kiwi_xl": obj["KIWI-XL"]["scores"],
                    }
    resultados = []
    for (_, ingles), dados in frases_dict.items():
        resultados.append({
            "indice": dados["indice"],
            "ingles_original": ingles,
            "portugues_traduzido": dados["portugues_traduzido"],
            "label": dados["label"],
            "dataset": dados["dataset"],
            "modelo": dados["modelo"],
            "prompt": dados["prompt"],
            "score_combinado": round(dados["score_combinado"], 4),
            "xcomet_xl": dados["xcomet_xl"],
            "comet22": dados["comet22"],
            "kiwi_xl": dados["kiwi_xl"],
        })
    return pd.DataFrame(resultados)

def selecionar_melhor_xcomet(dataset_nome):
    frases_dict = {}
    for modelo in MODELOS:
        prompts = PROMPTS_POR_MODELO[modelo]
        for prompt in prompts:
            if prompt:
                metricas_path = f"dataset_{dataset_nome}/{modelo}/{prompt}/frases_traduzidas_com_metricas.json"
            else:
                metricas_path = f"dataset_{dataset_nome}/{modelo}/frases_traduzidas_com_metricas.json"

            with open(metricas_path, "r") as f:
                dados = json.load(f)

            for obj in dados:
                ingles = obj["ingles_original"]
                xcomet = obj["XCOMET-XL"]["scores"]
                label = obj["label"]
                chave_frase = (label, ingles)
                portugues = obj["portugues_traduzido"]
                if chave_frase not in frases_dict or xcomet > frases_dict[chave_frase]["xcomet_xl"]:
                    frases_dict[chave_frase] = {
                        "ingles_original": ingles,
                        "portugues_traduzido": portugues,
                        "label": label,
                        "dataset": dataset_nome,
                        "modelo": modelo,
                        "prompt": prompt if prompt else "sem_prompt",
                        "xcomet_xl": xcomet,
                    }

    resultados = []
    for idx, ((_, ingles), dados) in enumerate(frases_dict.items()):
        resultados.append({
            "indice": idx,
            "ingles_original": ingles,
            "portugues_traduzido": dados["portugues_traduzido"],
            "label": dados["label"],
            "dataset": dados["dataset"],
            "modelo": dados["modelo"],
            "prompt": dados["prompt"],
            "xcomet_xl": round(dados["xcomet_xl"], 4),
        })
    return pd.DataFrame(resultados)

def aplicar_quartis(df):
    scores = df["xcomet_xl"]
    q1 = scores.quantile(0.25)
    q2 = scores.quantile(0.50)
    q3 = scores.quantile(0.75)

    df["quartil"] = pd.cut(
        scores,
        bins=[-np.inf, q1, q2, q3, np.inf],
        labels=QUARTIS_LABELS,
    )
    return df, {"q1": round(q1, 4), "q2": round(q2, 4), "q3": round(q3, 4)}

def gerar_datasets_finais_por_label(df, dataset_nome, saida_dir):
    score_col = "score_combinado"

    for tipo_label, nome_label, pasta in [(0, "literais", "nao_metaforicos"),(1, "metaforicas", "metaforicos")]:
        df_label = df[df["label"] == tipo_label].copy()
        if df_label.empty:
            print(f"  [AVISO] Nenhuma frase com label={tipo_label} ({nome_label}) em {dataset_nome}")
            continue

        df_label = df_label.sort_values(score_col, ascending=False).reset_index(drop=True)
        # Aplicar quartis na coluna correta
        if score_col == "score_combinado":
            scores = df_label[score_col]
            q1 = scores.quantile(0.25)
            q2 = scores.quantile(0.50)
            q3 = scores.quantile(0.75)
            df_label["quartil"] = pd.cut(
                scores,
                bins=[-np.inf, q1, q2, q3, np.inf],
                labels=QUARTIS_LABELS,
                duplicates='drop'
            )
            limites = {"q1": round(q1, 4), "q2": round(q2, 4), "q3": round(q3, 4)}
        else:
            df_label, limites = aplicar_quartis(df_label)
        print(f"  Total de frases {nome_label}: {len(df_label)}")
        for label in reversed(QUARTIS_LABELS):
            grupo = df_label[df_label["quartil"] == label][score_col]
            if not grupo.empty:
                print(f"    {label}: {len(grupo)} frases | max={grupo.max():.4f} min={grupo.min():.4f}")

        quartis_selecionados = QUARTIS_LABELS[-N_QUARTIS_SUPERIORES:]
        dataset_final = df_label[df_label["quartil"].isin(quartis_selecionados)].copy()
        dataset_final = dataset_final.drop_duplicates(subset=["ingles_original"], keep="first").reset_index(drop=True)

        # Ordenar por indice e, em caso de empate, por modelo e prompt conforme PROMPTS_POR_MODELO (sem hardcode)
        dataset_final = ordenar_por_indice(dataset_final)
        if "label" in dataset_final.columns:
            dataset_final = dataset_final.drop(columns=["label"])

        pasta_saida = os.path.join(saida_dir, pasta)
        os.makedirs(pasta_saida, exist_ok=True)
        saida_final = os.path.join(pasta_saida, "dataset_final.csv")
        dataset_final.to_csv(saida_final, index=False)
        print(f"  {os.path.join(pasta, 'dataset_final.csv')}: {len(dataset_final)} frases (quartis: {quartis_selecionados})\n")

def calcular_parascore_dataset_final(dataset_nome):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    saida_dir = f"dataset_{dataset_nome}/[CSV] melhor_xcomet"

    for tipo_label, subpasta in [("literais", "nao_metaforicos"), ("metaforicas", "metaforicos")]:
        pasta_label = os.path.join(saida_dir, subpasta)
        arquivo_final = os.path.join(pasta_label, "dataset_final.csv")
        if not os.path.exists(arquivo_final):
            print(f"[ERRO] dataset_final não encontrado para {dataset_nome} em {subpasta}.")
            continue

        df_final = pd.read_csv(arquivo_final)

        # Carregar todas as versões
        versoes = {}
        for modelo in MODELOS:
            for prompt in PROMPTS_POR_MODELO.get(modelo, []):
                if prompt:
                    caminho = f"dataset_{dataset_nome}/{modelo}/{prompt}/matriz.csv"
                else:
                    caminho = f"dataset_{dataset_nome}/{modelo}/matriz.csv"
                dados = pd.read_csv(caminho)
                versoes[(modelo, prompt)] = dados

        scorer = ParaScorer(lang="pt", model_type="xlm-roberta-base")

        registros = []
        print(f"\nCalculando paraScore para {dataset_nome} ({tipo_label}) ({len(df_final)} frases)...")
        for _, row in df_final.iterrows():
            idx = row["indice"]
            melhor_trad = row["portugues_traduzido"]

            if pd.isna(melhor_trad) or isinstance(melhor_trad, float):
                melhor_trad = "[vazio]"
            else:
                melhor_trad = str(melhor_trad).strip()
            modelo_melhor = row["modelo"]
            prompt_melhor = row["prompt"]

            for (modelo, prompt), dados in versoes.items():
                prompt_label = prompt if prompt is not None else "sem_prompt"
                if modelo == modelo_melhor and prompt_label == prompt_melhor:
                    continue

                outra_trad = dados.iloc[idx]["portugues_traduzido"]
                if pd.isna(outra_trad) or isinstance(outra_trad, float):
                    outra_trad = "[vazio]"
                else:
                    outra_trad = str(outra_trad).strip()
                if not outra_trad or outra_trad.lower() == "nan":
                    outra_trad = "[vazio]"

                score = scorer.free_score([outra_trad], [melhor_trad], batch_size=16)[0]

                registros.append({
                    "indice": idx,
                    "ingles_original": row["ingles_original"],
                    "melhor_trad": melhor_trad,
                    "melhor_modelo": modelo_melhor,
                    "melhor_prompt": prompt_melhor,
                    "outro_modelo": modelo,
                    "outro_prompt": prompt_label,
                    "outra_trad": outra_trad,
                    "parascore": round(float(score), 4),
                })

        saida = os.path.join(pasta_label, "parascore.csv")
        df_registros = pd.DataFrame(registros).sort_values(
            by=["parascore", "indice"],
            ascending=[False, True]
        ).reset_index(drop=True)
        colunas_parascore = [
            "indice",
            "ingles_original",
            "melhor_trad",
            "outra_trad",
            "melhor_modelo",
            "melhor_prompt",
            "outro_modelo",
            "outro_prompt",
            "parascore",
        ]
        df_registros = df_registros[colunas_parascore]
        df_registros.to_csv(saida, index=False)
        print(f"  {len(registros)} comparações salvas em: {os.path.join(subpasta, 'parascore.csv')}")

def plotar_distribuicao_parascore(dataset_nome):
    base_dir = f"dataset_{dataset_nome}"
    saida_dir = os.path.join(base_dir, "[CSV] melhor_xcomet")
    for tipo_label, subpasta in [("literais", "nao_metaforicos"), ("metaforicas", "metaforicos")]:
        pasta_label = os.path.join(saida_dir, subpasta)
        arquivo = os.path.join(pasta_label, "parascore.csv")
        if not os.path.exists(arquivo):
            print(f"[AVISO] parascore não encontrado para {dataset_nome} ({tipo_label}), pulando plot.")
            continue

        df = pd.read_csv(arquivo)
        scores = df["parascore"].dropna()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(scores, bins=40, kde=True, ax=ax, color="#4C72B0", edgecolor="white", alpha=0.75)
        ax.axvline(scores.mean(),   color="black", linestyle="--", linewidth=1.2, label=f"Média: {scores.mean():.3f}")
        ax.axvline(scores.median(), color="gray",  linestyle=":",  linewidth=1.2, label=f"Mediana: {scores.median():.3f}")

        ax.set_title(f"Distribuição do ParaScore — {dataset_nome} ({tipo_label})", fontsize=13, fontweight="bold")
        ax.set_xlabel("ParaScore", fontsize=11)
        ax.set_ylabel("Frequência", fontsize=11)
        ax.legend(fontsize=9)

        saida_png = os.path.join(pasta_label, "distribuicao_parascore.png")
        plt.tight_layout()
        plt.savefig(saida_png)
        plt.close()
        print(f"  Distribuição salva em: {os.path.join(subpasta, 'distribuicao_parascore.png')}")

def plotar_distribuicao_score_combinado(dataset_nome):
    base_dir = f"dataset_{dataset_nome}"
    saida_dir = os.path.join(base_dir, "[CSV] melhor_xcomet")

    for tipo_label, subpasta in [(0, "nao_metaforicos"), (1, "metaforicos")]:
        pasta_label = os.path.join(saida_dir, subpasta)
        arquivo = os.path.join(pasta_label, "dataset_final.csv")
        if not os.path.exists(arquivo):
            print(f"[AVISO] dataset_final não encontrado para {dataset_nome} em {subpasta}")
            continue

        df = pd.read_csv(arquivo)
        if df.empty or "score_combinado" not in df.columns:
            print(f"[AVISO] Sem dados de score_combinado para {subpasta} em {dataset_nome}")
            continue

        scores = df["score_combinado"].dropna()
        if (scores < 0).any() or (scores > 1).any():
            print(f"[ALERTA] Existem valores de score_combinado fora do intervalo [0, 1] em {subpasta} ({dataset_nome}). Exemplo de valores: {scores[(scores < 0) | (scores > 1)].head(5).tolist()}")

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(scores, bins=40, kde=True, ax=ax, color="#55A868", edgecolor="white", alpha=0.75)
        ax.axvline(scores.mean(),   color="black", linestyle="--", linewidth=1.2, label=f"Média: {scores.mean():.3f}")
        ax.axvline(scores.median(), color="gray",  linestyle=":",  linewidth=1.2, label=f"Mediana: {scores.median():.3f}")
        ax.set_title(f"Distribuição do Score Combinado — {dataset_nome} ({subpasta})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Score Combinado", fontsize=11)
        ax.set_ylabel("Frequência", fontsize=11)
        ax.legend(fontsize=9)

        os.makedirs(pasta_label, exist_ok=True)
        saida_png = os.path.join(pasta_label, "distribuicao_score_combinado.png")
        plt.tight_layout()
        plt.savefig(saida_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [{dataset_nome}] {subpasta}: média score_combinado={scores.mean():.4f} | mediana={scores.median():.4f}")
        print(f"  Distribuição salva em: {os.path.join(subpasta, 'distribuicao_score_combinado.png')}")

if __name__ == "__main__":

    for dataset_nome in DATASETS:
        processar_dataset(dataset_nome)
