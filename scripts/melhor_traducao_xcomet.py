import os
import json
import pandas as pd
import numpy as np
from parascore import ParaScorer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

MODELOS = ["gpt", "gemini", "gemma3", "gemmaX"]
PROMPTS_POR_MODELO = {
    "gpt": ["prompt2", "prompt4"],
    "gemini": ["prompt2", "prompt4"],
    "gemma3": ["prompt2", "prompt4"],
    "gemmaX": [None],  # sem prompt (modelo de tradução)
}
DATASETS = ["newsmet", "manual_data"]
QUARTIS_LABELS = ["Q4", "Q3", "Q2", "Q1"]  # do pior para o melhor
N_QUARTIS_SUPERIORES = 2  # quantos dos melhores quartis incluir no dataset_final

def carregar_json_metricas(base_dir, modelo, prompt):
    if prompt:
        caminho = os.path.join(base_dir, modelo, prompt, "frases_traduzidas_com_metricas.json")
    else:
        caminho = os.path.join(base_dir, modelo, "frases_traduzidas_com_metricas.json")
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def selecionar_melhor_xcomet(dataset_nome):
    base_dir = os.path.join(BASE_DIR, f"dataset_{dataset_nome}")

    # Carregar todas as 7 versões
    versoes = {}
    for modelo in MODELOS:
        for prompt in PROMPTS_POR_MODELO.get(modelo, []):
            try:
                dados = carregar_json_metricas(base_dir, modelo, prompt)
                versoes[(modelo, prompt)] = dados
            except FileNotFoundError:
                prompt_desc = prompt if prompt else "sem_prompt"
                print(f"[AVISO] Arquivo não encontrado: {modelo}/{prompt_desc} em {dataset_nome}")

    if not versoes:
        print(f"Nenhuma versão encontrada para {dataset_nome}")
        return None

    # Assumir que todas as versões têm o mesmo número de frases e mesma ordem
    primeira_chave = list(versoes.keys())[0]
    num_frases = len(versoes[primeira_chave])

    resultados = []
    for idx in range(num_frases):
        melhor_score = -float("inf")
        melhor_registro = None

        for (modelo, prompt), dados in versoes.items():
            item = dados[idx]
            xcomet_score = item["XCOMET-XL"]["scores"]
            kiwi_score   = item["KIWI-XL"]["scores"]
            comet_score  = item["COMET22"]["scores"]
            score_combinado =  (xcomet_score + kiwi_score + comet_score) / 3

            if score_combinado > melhor_score:
                melhor_score = score_combinado
                prompt_label = prompt if prompt is not None else "sem_prompt"
                melhor_registro = {
                    "indice": idx,
                    "dataset": dataset_nome,
                    "ingles_original": item["ingles_original"],
                    "portugues_traduzido": item["portugues_traduzido"],
                    "modelo": modelo,
                    "prompt": prompt_label,
                    "xcomet_xl": round(xcomet_score, 4),
                    "kiwi_xl": round(kiwi_score, 4),
                    "comet22": round(comet_score, 4),
                    "score_combinado": round(score_combinado, 4),
                }

        if melhor_registro:
            resultados.append(melhor_registro)

    return pd.DataFrame(resultados)


def aplicar_quartis(df):
    scores = df["score_combinado"]
    q1 = scores.quantile(0.25)
    q2 = scores.quantile(0.50)
    q3 = scores.quantile(0.75)

    # Score combinado mais alto = melhor tradução → Q1 = top 25%
    df["quartil"] = pd.cut(
        scores,
        bins=[-np.inf, q1, q2, q3, np.inf],
        labels=QUARTIS_LABELS,
    )
    return df, {"q1": round(q1, 4), "q2": round(q2, 4), "q3": round(q3, 4)}


def calcular_parascore_dataset_final(dataset_nome):
    logging.getLogger("transformers").setLevel(logging.ERROR)

    saida_dir = os.path.join(BASE_DIR, f"dataset_{dataset_nome}", "[CSV] melhor_xcomet")
    arquivo_final = os.path.join(saida_dir, f"dataset_final_{dataset_nome}.csv")

    if not os.path.exists(arquivo_final):
        print(f"[ERRO] dataset_final não encontrado para {dataset_nome}. Rode o script sem flags primeiro.")
        return

    df_final = pd.read_csv(arquivo_final)
    base_dir = os.path.join(BASE_DIR, f"dataset_{dataset_nome}")

    # Carregar todas as versões
    versoes = {}
    for modelo in MODELOS:
        for prompt in PROMPTS_POR_MODELO.get(modelo, []):
            try:
                dados = carregar_json_metricas(base_dir, modelo, prompt)
                versoes[(modelo, prompt)] = dados
            except FileNotFoundError:
                pass

    scorer = ParaScorer(lang="pt", model_type="xlm-roberta-base")

    registros = []
    print(f"\nCalculando paraScore para {dataset_nome} ({len(df_final)} frases)...")
    for _, row in df_final.iterrows():
        idx = row["indice"]
        melhor_trad = row["portugues_traduzido"]
        modelo_melhor = row["modelo"]
        prompt_melhor = row["prompt"]

        for (modelo, prompt), dados in versoes.items():
            prompt_label = prompt if prompt is not None else "sem_prompt"
            # Pular a própria versão selecionada como melhor
            if modelo == modelo_melhor and prompt_label == prompt_melhor:
                continue

            outra_trad = str(dados[idx]["portugues_traduzido"]).strip()
            if not outra_trad or outra_trad.lower() == "nan":
                outra_trad = "[vazio]"

            score = scorer.free_score([outra_trad], [melhor_trad], batch_size=16)[0]

            registros.append({
                "indice": idx,
                "dataset": dataset_nome,
                "ingles_original": row["ingles_original"],
                "melhor_trad": melhor_trad,
                "melhor_modelo": modelo_melhor,
                "melhor_prompt": prompt_melhor,
                "outro_modelo": modelo,
                "outro_prompt": prompt_label,
                "outra_trad": outra_trad,
                "parascore": round(float(score), 4),
            })

    saida = os.path.join(saida_dir, f"parascore_{dataset_nome}.csv")
    pd.DataFrame(registros).to_csv(saida, index=False)
    print(f"  {len(registros)} comparações salvas em: parascore_{dataset_nome}.csv")


def plotar_distribuicao_parascore(dataset_nome):

    saida_dir = os.path.join(BASE_DIR, f"dataset_{dataset_nome}", "[CSV] melhor_xcomet")
    arquivo = os.path.join(saida_dir, f"parascore_{dataset_nome}.csv")

    if not os.path.exists(arquivo):
        print(f"[AVISO] parascore não encontrado para {dataset_nome}, pulando plot.")
        return

    df = pd.read_csv(arquivo)
    scores = df["parascore"].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(scores, bins=40, kde=True, ax=ax, color="#4C72B0", edgecolor="white", alpha=0.75)
    ax.axvline(scores.mean(),   color="black", linestyle="--", linewidth=1.2, label=f"Média: {scores.mean():.3f}")
    ax.axvline(scores.median(), color="gray",  linestyle=":",  linewidth=1.2, label=f"Mediana: {scores.median():.3f}")

    ax.set_title(f"Distribuição do ParaScore — {dataset_nome}", fontsize=13, fontweight="bold")
    ax.set_xlabel("ParaScore", fontsize=11)
    ax.set_ylabel("Frequência", fontsize=11)
    ax.legend(fontsize=9)

    saida_png = os.path.join(saida_dir, f"distribuicao_parascore_{dataset_nome}.png")
    plt.tight_layout()
    plt.savefig(saida_png, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Distribuição salva em: distribuicao_parascore_{dataset_nome}.png")


def plotar_distribuicao_xcomet(dataset_nome):
    base_dir = os.path.join(BASE_DIR, f"dataset_{dataset_nome}")

    scores_por_frase = []
    versoes = {}
    for modelo in MODELOS:
        for prompt in PROMPTS_POR_MODELO.get(modelo, []):
            try:
                dados = carregar_json_metricas(base_dir, modelo, prompt)
                versoes[(modelo, prompt)] = dados
            except FileNotFoundError:
                pass

    if not versoes:
        print(f"[AVISO] Nenhuma versão encontrada para {dataset_nome}")
        return

    num_frases = len(list(versoes.values())[0])
    for idx in range(num_frases):
        melhor = max(
            dados[idx]["XCOMET-XL"]["scores"]
            for dados in versoes.values()
        )
        scores_por_frase.append(melhor)

    scores = pd.Series(scores_por_frase)
    iguais_1 = (scores >= 0.9999).sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(scores, bins=40, kde=True, ax=ax, color="#55A868", edgecolor="white", alpha=0.75)
    ax.axvline(scores.mean(),   color="black", linestyle="--", linewidth=1.2, label=f"Média: {scores.mean():.3f}")
    ax.axvline(scores.median(), color="gray",  linestyle=":",  linewidth=1.2, label=f"Mediana: {scores.median():.3f}")
    ax.set_title(f"Distribuição do melhor XCOMET-XL por frase — {dataset_nome}\n(scores=1.0: {iguais_1}/{len(scores)} = {iguais_1/len(scores)*100:.1f}%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("XCOMET-XL", fontsize=11)
    ax.set_ylabel("Frequência", fontsize=11)
    ax.legend(fontsize=9)

    saida_dir = os.path.join(BASE_DIR, f"dataset_{dataset_nome}", "[CSV] melhor_xcomet")
    os.makedirs(saida_dir, exist_ok=True)
    saida_png = os.path.join(saida_dir, f"distribuicao_xcomet_{dataset_nome}.png")
    plt.tight_layout()
    plt.savefig(saida_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [{dataset_nome}] Frases com XCOMET-XL=1.0: {iguais_1}/{len(scores)} ({iguais_1/len(scores)*100:.1f}%)")
    print(f"  Distribuição salva em: distribuicao_xcomet_{dataset_nome}.png")


def plotar_distribuicao_combinado(dataset_nome, df):
    scores = df["score_combinado"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(scores, bins=40, kde=True, ax=ax, color="#C44E52", edgecolor="white", alpha=0.75)
    ax.axvline(scores.mean(),   color="black", linestyle="--", linewidth=1.2, label=f"Média: {scores.mean():.3f}")
    ax.axvline(scores.median(), color="gray",  linestyle=":",  linewidth=1.2, label=f"Mediana: {scores.median():.3f}")
    ax.set_title(f"Distribuição do Score Combinado (média de XCOMET-XL, KIWI-XL e COMET22) — {dataset_nome}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Score Combinado", fontsize=11)
    ax.set_ylabel("Frequência", fontsize=11)
    ax.legend(fontsize=9)

    saida_dir = os.path.join(BASE_DIR, f"dataset_{dataset_nome}", "[CSV] melhor_xcomet")
    os.makedirs(saida_dir, exist_ok=True)
    saida_png = os.path.join(saida_dir, f"distribuicao_combinado_{dataset_nome}.png")
    plt.tight_layout()
    plt.savefig(saida_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [{dataset_nome}] Score combinado — min={scores.min():.3f} max={scores.max():.3f} std={scores.std():.3f}")
    print(f"  Distribuição salva em: distribuicao_combinado_{dataset_nome}.png")


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "--plotar-parascore":
        # python scripts/melhor_traducao_xcomet.py --plotar-parascore
        for dataset_nome in DATASETS:
            plotar_distribuicao_parascore(dataset_nome)
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--plotar-xcomet":
        # python scripts/melhor_traducao_xcomet.py --plotar-xcomet
        for dataset_nome in DATASETS:
            plotar_distribuicao_xcomet(dataset_nome)
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--plotar-combinado":
        # python scripts/melhor_traducao_xcomet.py --plotar-combinado
        for dataset_nome in DATASETS:
            df = selecionar_melhor_xcomet(dataset_nome)
            if df is not None and not df.empty:
                plotar_distribuicao_combinado(dataset_nome, df)
        sys.exit(0)

    for dataset_nome in DATASETS:
        print(f"\nProcessando dataset: {dataset_nome}")
        df = selecionar_melhor_xcomet(dataset_nome)
        if df is None or df.empty:
            continue

        df, limites = aplicar_quartis(df)

        print(f"  Total de frases: {len(df)}")
        for label in reversed(QUARTIS_LABELS):
            grupo = df[df["quartil"] == label]["score_combinado"]
            if not grupo.empty:
                print(f"    {label}: {len(grupo)} frases | max={grupo.max():.4f} min={grupo.min():.4f}")

        # Salvar CSV por dataset
        saida_dir = os.path.join(BASE_DIR, f"dataset_{dataset_nome}", "[CSV] melhor_xcomet")
        os.makedirs(saida_dir, exist_ok=True)

        saida = os.path.join(saida_dir, f"melhor_xcomet_{dataset_nome}.csv")
        df_ordenado = df.sort_values("score_combinado", ascending=False).reset_index(drop=True)
        df_ordenado.to_csv(saida, index=False)

        # Frases dos N quartis superiores vão para o dataset_final (controlado por N_QUARTIS_SUPERIORES)
        quartis_selecionados = QUARTIS_LABELS[-N_QUARTIS_SUPERIORES:]
        dataset_final = df_ordenado[df_ordenado["quartil"].isin(quartis_selecionados)].copy()
        dataset_final = dataset_final.reset_index(drop=True)

        saida_final = os.path.join(saida_dir, f"dataset_final_{dataset_nome}.csv")
        dataset_final.to_csv(saida_final, index=False)
        print(f"  dataset_final: {len(dataset_final)} frases (quartis: {quartis_selecionados})\n")

        calcular_parascore_dataset_final(dataset_nome)
        plotar_distribuicao_parascore(dataset_nome)
