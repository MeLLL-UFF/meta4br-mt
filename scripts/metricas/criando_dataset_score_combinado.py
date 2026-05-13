import os
import sys
import pandas as pd
import numpy as np
import json
from parascore import ParaScorer
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Modelos e prompts ja selecionados, os melhores
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
PASTA_SAIDA = "[CSV] melhor_score_combinado"
ARQUIVO_MELHOR_SCORE = "melhor_score_combinado.csv"
LIMIAR_CONJUNTO_LIBERAL = 0.9
LIMIAR_CONJUNTO_CONSERVADOR = 0.95


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
        return len(ordem_modelos)


def get_prompt_idx(modelo, prompt):
    ordem_prompts = PROMPTS_POR_MODELO.get(modelo, [])
    if prompt == "sem_prompt":
        if None in ordem_prompts:
            return ordem_prompts.index(None)
        return len(ordem_prompts)
    try:
        return ordem_prompts.index(prompt)
    except ValueError:
        return len(ordem_prompts)


def processar_dataset(dataset_nome):
    print(f"\nProcessando dataset: {dataset_nome}")
    print("Selecionando melhores frases por score combinado de COMET22, XCOMET-XL e KIWI-XL")
    df = selecionar_melhor_score_combinado(dataset_nome)
    if df is None or df.empty:
        print(f"  [AVISO] Nenhuma frase encontrada para {dataset_nome}")
        return

    saida_dir = os.path.join(f"dataset_{dataset_nome}", PASTA_SAIDA)
    os.makedirs(saida_dir, exist_ok=True)

    saida = os.path.join(saida_dir, ARQUIVO_MELHOR_SCORE)
    df_ordenado = ordenar_por_indice(df.copy())
    df_ordenado.to_csv(saida, index=False)

    gerar_datasets_finais_por_label(df, dataset_nome, saida_dir)
    calcular_parascore_dataset_final(dataset_nome)
    gerar_conjuntos_por_parascore(dataset_nome)
    plotar_distribuicao_parascore(dataset_nome)
    plotar_distribuicao_score_combinado(dataset_nome)


def selecionar_melhor_score_combinado(dataset_nome):
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


def aplicar_quartis(df, score_col):
    scores = df[score_col]
    q1 = scores.quantile(0.25)
    q2 = scores.quantile(0.50)
    q3 = scores.quantile(0.75)

    df["quartil"] = pd.cut(
        scores,
        bins=[-np.inf, q1, q2, q3, np.inf],
        labels=QUARTIS_LABELS,
        duplicates="drop",
    )
    return df, {"q1": round(q1, 4), "q2": round(q2, 4), "q3": round(q3, 4)}


def gerar_datasets_finais_por_label(df, dataset_nome, saida_dir):
    score_col = "score_combinado"

    for tipo_label, nome_label, pasta in [(0, "literais", "nao_metaforicos"), (1, "metaforicas", "metaforicos")]:
        df_label = df[df["label"] == tipo_label].copy()
        if df_label.empty:
            print(f"  [AVISO] Nenhuma frase com label={tipo_label} ({nome_label}) em {dataset_nome}")
            continue

        df_label = df_label.sort_values(score_col, ascending=False).reset_index(drop=True)
        df_label, limites = aplicar_quartis(df_label, score_col)
        print(f"  Total de frases {nome_label}: {len(df_label)}")
        print(
            "    Limites dos quartis de score_combinado: "
            f"q1={limites['q1']:.4f} | q2={limites['q2']:.4f} | q3={limites['q3']:.4f}"
        )
        for label in reversed(QUARTIS_LABELS):
            grupo = df_label[df_label["quartil"] == label][score_col]
            if not grupo.empty:
                print(f"    {label}: {len(grupo)} frases | max={grupo.max():.4f} min={grupo.min():.4f}")

        quartis_selecionados = QUARTIS_LABELS[-N_QUARTIS_SUPERIORES:]
        dataset_final = df_label[df_label["quartil"].isin(quartis_selecionados)].copy()
        dataset_final = dataset_final.drop_duplicates(subset=["ingles_original"], keep="first").reset_index(drop=True)

        dataset_final = ordenar_por_indice(dataset_final)
        colunas_para_remover = [coluna for coluna in ["label", "dataset"] if coluna in dataset_final.columns]
        if colunas_para_remover:
            dataset_final = dataset_final.drop(columns=colunas_para_remover)

        pasta_saida = os.path.join(saida_dir, pasta)
        os.makedirs(pasta_saida, exist_ok=True)
        saida_final = os.path.join(pasta_saida, "dataset_final.csv")
        dataset_final.to_csv(saida_final, index=False)
        print(f"  {os.path.join(pasta, 'dataset_final.csv')}: {len(dataset_final)} frases (quartis: {quartis_selecionados})\n")


def carregar_metricas_por_versao(dataset_nome):
    metricas_por_versao = {}
    for modelo in MODELOS:
        for prompt in PROMPTS_POR_MODELO.get(modelo, []):
            if prompt:
                caminho = f"dataset_{dataset_nome}/{modelo}/{prompt}/frases_traduzidas_com_metricas.json"
            else:
                caminho = f"dataset_{dataset_nome}/{modelo}/frases_traduzidas_com_metricas.json"

            with open(caminho, "r") as arquivo:
                dados = json.load(arquivo)

            prompt_label = prompt if prompt is not None else "sem_prompt"
            metricas_por_versao[(modelo, prompt_label)] = dados
    return metricas_por_versao


def montar_registro_de_metricas(dataset_nome, registro_metricas, indice, modelo, prompt, quartil=None, origem="parascore", parascore=None):
    xcomet = float(registro_metricas["XCOMET-XL"]["scores"])
    comet22 = float(registro_metricas["COMET22"]["scores"])
    kiwi_xl = float(registro_metricas["KIWI-XL"]["scores"])

    return {
        "indice": indice,
        "ingles_original": registro_metricas["ingles_original"],
        "portugues_traduzido": registro_metricas["portugues_traduzido"],
        "dataset": dataset_nome,
        "modelo": modelo,
        "prompt": prompt,
        "score_combinado": round((comet22 + xcomet + kiwi_xl) / 3.0, 4),
        "xcomet_xl": xcomet,
        "comet22": comet22,
        "kiwi_xl": kiwi_xl,
        "quartil": quartil,
        "origem": origem,
        "parascore": parascore,
    }


def gerar_conjuntos_por_parascore(dataset_nome):
    saida_dir = f"dataset_{dataset_nome}/{PASTA_SAIDA}"
    metricas_por_versao = carregar_metricas_por_versao(dataset_nome)
    configuracoes = [
        ("dataset_final_liberal.csv", LIMIAR_CONJUNTO_LIBERAL),
        ("dataset_final_conservador.csv", LIMIAR_CONJUNTO_CONSERVADOR),
    ]

    for _, subpasta in [("literais", "nao_metaforicos"), ("metaforicas", "metaforicos")]:
        pasta_label = os.path.join(saida_dir, subpasta)
        arquivo_final = os.path.join(pasta_label, "dataset_final.csv")
        arquivo_parascore = os.path.join(pasta_label, "parascore.csv")
        if not os.path.exists(arquivo_final) or not os.path.exists(arquivo_parascore):
            print(f"[AVISO] dataset_final ou parascore ausente para {dataset_nome} em {subpasta}, pulando conjuntos por limiar.")
            continue

        df_final = pd.read_csv(arquivo_final)
        df_parascore = pd.read_csv(arquivo_parascore)

        if "dataset" in df_final.columns:
            df_final = df_final.drop(columns=["dataset"])

        registros_base = df_final.copy()
        registros_base["origem"] = "dataset_final"
        registros_base["parascore"] = np.nan

        for nome_arquivo, limiar in configuracoes:
            selecionados = df_parascore[df_parascore["parascore"] >= limiar].copy()
            registros_adicionados = []

            for _, row in selecionados.iterrows():
                chave_versao = (row["outro_modelo"], row["outro_prompt"])
                registros_da_versao = metricas_por_versao.get(chave_versao)
                if registros_da_versao is None:
                    continue

                indice = int(row["indice"])
                registro_metricas = registros_da_versao[indice]
                quartil = df_final.loc[df_final["indice"] == indice, "quartil"]
                quartil = quartil.iloc[0] if not quartil.empty else np.nan

                registros_adicionados.append(
                    montar_registro_de_metricas(
                        dataset_nome=dataset_nome,
                        registro_metricas=registro_metricas,
                        indice=indice,
                        modelo=row["outro_modelo"],
                        prompt=row["outro_prompt"],
                        quartil=quartil,
                        origem="parascore",
                        parascore=round(float(row["parascore"]), 4),
                    )
                )

            df_extra = pd.DataFrame(registros_adicionados)
            if not df_extra.empty:
                df_extra = df_extra[registros_base.columns]

            conjunto = pd.concat([registros_base, df_extra], ignore_index=True)
            conjunto = conjunto.assign(
                _texto_normalizado=conjunto["portugues_traduzido"].fillna("").astype(str).str.strip()
            )
            conjunto = conjunto.drop_duplicates(subset=["_texto_normalizado"], keep="first")
            conjunto = conjunto.drop(columns=["_texto_normalizado"]).reset_index(drop=True)
            conjunto = ordenar_por_indice(conjunto)
            saida_conjunto = os.path.join(pasta_label, nome_arquivo)
            conjunto.to_csv(saida_conjunto, index=False)
            total_frases = len(conjunto)
            print(
                f"  {os.path.join(subpasta, nome_arquivo)}: {total_frases} frases (limiar parascore >= {limiar})"
            )


def parse_args(argv):
    if not argv:
        return "rodar-tudo"

    argumentos_validos = {"--rodar-tudo", "--rodar-threshold"}
    argumentos_recebidos = set(argv)

    if not argumentos_recebidos.issubset(argumentos_validos):
        desconhecidos = [arg for arg in argv if arg not in argumentos_validos]
        raise ValueError(f"Argumento desconhecido: {', '.join(desconhecidos)}")

    if "--rodar-tudo" in argumentos_recebidos and "--rodar-threshold" in argumentos_recebidos:
        raise ValueError("Use apenas um modo por vez: --rodar-tudo ou --rodar-threshold")

    if "--rodar-threshold" in argumentos_recebidos:
        return "rodar-threshold"

    return "rodar-tudo"


def calcular_parascore_dataset_final(dataset_nome):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    saida_dir = f"dataset_{dataset_nome}/{PASTA_SAIDA}"

    for tipo_label, subpasta in [("literais", "nao_metaforicos"), ("metaforicas", "metaforicos")]:
        pasta_label = os.path.join(saida_dir, subpasta)
        arquivo_final = os.path.join(pasta_label, "dataset_final.csv")
        if not os.path.exists(arquivo_final):
            print(f"[ERRO] dataset_final nao encontrado para {dataset_nome} em {subpasta}.")
            continue

        df_final = pd.read_csv(arquivo_final)

        metricas_por_versao = carregar_metricas_por_versao(dataset_nome)

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

            for (modelo, prompt_label), registros_da_versao in metricas_por_versao.items():
                if modelo == modelo_melhor and prompt_label == prompt_melhor:
                    continue

                outra_trad = registros_da_versao[idx]["portugues_traduzido"]
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
        print(f"  {len(registros)} comparacoes salvas em: {os.path.join(subpasta, 'parascore.csv')}")


def plotar_distribuicao_parascore(dataset_nome):
    base_dir = f"dataset_{dataset_nome}"
    saida_dir = os.path.join(base_dir, PASTA_SAIDA)
    for tipo_label, subpasta in [("literais", "nao_metaforicos"), ("metaforicas", "metaforicos")]:
        pasta_label = os.path.join(saida_dir, subpasta)
        arquivo = os.path.join(pasta_label, "parascore.csv")
        if not os.path.exists(arquivo):
            print(f"[AVISO] parascore nao encontrado para {dataset_nome} ({tipo_label}), pulando plot.")
            continue

        df = pd.read_csv(arquivo)
        scores = df["parascore"].dropna()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(scores, bins=40, kde=True, ax=ax, color="#4C72B0", edgecolor="white", alpha=0.75)
        ax.axvline(scores.mean(), color="black", linestyle="--", linewidth=1.2, label=f"Media: {scores.mean():.3f}")
        ax.axvline(scores.median(), color="gray", linestyle=":", linewidth=1.2, label=f"Mediana: {scores.median():.3f}")

        ax.set_title(f"Distribuicao do ParaScore - {dataset_nome} ({tipo_label})", fontsize=13, fontweight="bold")
        ax.set_xlabel("ParaScore", fontsize=11)
        ax.set_ylabel("Frequencia", fontsize=11)
        ax.legend(fontsize=9)

        saida_png = os.path.join(pasta_label, "distribuicao_parascore.png")
        plt.tight_layout()
        plt.savefig(saida_png)
        plt.close()
        print(f"  Distribuicao salva em: {os.path.join(subpasta, 'distribuicao_parascore.png')}")


def plotar_distribuicao_score_combinado(dataset_nome):
    base_dir = f"dataset_{dataset_nome}"
    saida_dir = os.path.join(base_dir, PASTA_SAIDA)

    for tipo_label, subpasta in [(0, "nao_metaforicos"), (1, "metaforicos")]:
        pasta_label = os.path.join(saida_dir, subpasta)
        arquivo = os.path.join(pasta_label, "dataset_final.csv")
        if not os.path.exists(arquivo):
            print(f"[AVISO] dataset_final nao encontrado para {dataset_nome} em {subpasta}")
            continue

        df = pd.read_csv(arquivo)
        if df.empty or "score_combinado" not in df.columns:
            print(f"[AVISO] Sem dados de score_combinado para {subpasta} em {dataset_nome}")
            continue

        scores = df["score_combinado"].dropna()
        if (scores < 0).any() or (scores > 1).any():
            print(
                f"[ALERTA] Existem valores de score_combinado fora do intervalo [0, 1] em {subpasta} ({dataset_nome}). "
                f"Exemplo de valores: {scores[(scores < 0) | (scores > 1)].head(5).tolist()}"
            )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(scores, bins=40, kde=True, ax=ax, color="#55A868", edgecolor="white", alpha=0.75)
        ax.axvline(scores.mean(), color="black", linestyle="--", linewidth=1.2, label=f"Media: {scores.mean():.3f}")
        ax.axvline(scores.median(), color="gray", linestyle=":", linewidth=1.2, label=f"Mediana: {scores.median():.3f}")
        ax.set_title(f"Distribuicao do Score Combinado - {dataset_nome} ({subpasta})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Score Combinado", fontsize=11)
        ax.set_ylabel("Frequencia", fontsize=11)
        ax.legend(fontsize=9)

        os.makedirs(pasta_label, exist_ok=True)
        saida_png = os.path.join(pasta_label, "distribuicao_score_combinado.png")
        plt.tight_layout()
        plt.savefig(saida_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [{dataset_nome}] {subpasta}: media score_combinado={scores.mean():.4f} | mediana={scores.median():.4f}")
        print(f"  Distribuicao salva em: {os.path.join(subpasta, 'distribuicao_score_combinado.png')}")


if __name__ == "__main__":
    try:
        modo_execucao = parse_args(sys.argv[1:])
    except ValueError as exc:
        print(f"[ERRO] {exc}")
        print(
            "Uso: python scripts/metricas/criando_dataset_score_combinado.py "
            "[--rodar-tudo | --rodar-threshold]"
        )
        sys.exit(1)

    for dataset_nome in DATASETS:
        if modo_execucao == "rodar-threshold":
            print(f"\nRegenerando conjuntos por parascore para: {dataset_nome}")
            gerar_conjuntos_por_parascore(dataset_nome)
        else:
            processar_dataset(dataset_nome)