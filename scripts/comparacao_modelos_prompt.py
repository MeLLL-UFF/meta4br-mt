import os
from itertools import combinations
import numpy as np
import pandas as pd
from rouge import Rouge as RougeLib
import sys
from tqdm import tqdm

# Esses comandos serão usados caso altere a anotação das frases como consistente ou discrepante
# python scripts/comparacao_modelos_prompt.py --calcular-ranking newsmet
# python scripts/comparacao_modelos_prompt.py --calcular-ranking manual_data

# Para rodar o rouge em cima das frases consistentes
# python scripts/comparacao_modelos_prompt.py --frases-consistentes newsmet
# python scripts/comparacao_modelos_prompt.py --frases-consistentes manual_data

os.environ["LD_LIBRARY_PATH"] = ""
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  

MODELOS = ["gpt", "gemini", "gemma3", "gemmaX"]
PROMPTS_POR_MODELO = {
    "gpt": ["prompt2", "prompt4"],
    "gemini": ["prompt2", "prompt4"],
    "gemma3": ["prompt2", "prompt4"],
    "gemmaX": [None],
}
DATASETS = {
    "newsmet": os.path.join(BASE_DIR, "dataset_newsmet"),
    "manual_data": os.path.join(BASE_DIR, "dataset_manual_data"),
}
VARIACAO_PERMITIDA = 0.25
BATCH_SALVAMENTO = 5
# LIMITE_FRASES = 20


def preparar_arquivo_saida(caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    if os.path.exists(caminho):
        os.remove(caminho)


def append_registros_csv(caminho, registros):
    if not registros:
        return
    df = pd.DataFrame(registros)
    escrever_cabecalho = not os.path.exists(caminho)
    df.to_csv(caminho, mode="a", header=escrever_cabecalho, index=False)


def carregar_traducoes(base_dir, modelo, prompt):
    # Ajuste pra o gemmaX que não tem prompt
    if prompt:
        caminho = os.path.join(base_dir, modelo, prompt, "matriz.csv")
    else:
        caminho = os.path.join(base_dir, modelo, "matriz.csv")
    df = pd.read_csv(caminho)

    colunas_necessarias = ["ingles_original", "portugues_traduzido"]
    for col in colunas_necessarias:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' nao encontrada em {caminho}")

    df = df[colunas_necessarias]
    return df


def analisar_consistencia(dataset_nome, base_dir):
   
    # Carregar todas as traduções
    traducoes = {}
    for modelo in MODELOS:
        for prompt in PROMPTS_POR_MODELO.get(modelo, []):
            try:
                df = carregar_traducoes(base_dir, modelo, prompt)
                traducoes[(modelo, prompt)] = df
            except FileNotFoundError:
                prompt_desc = prompt if prompt else "sem_prompt"
                print(f"Arquivo nao encontrado para {modelo}/{prompt_desc} em {dataset_nome}")

    if not traducoes:
        print(f"Nenhuma traducao encontrada para {dataset_nome}")
        return

    primeira_chave = list(traducoes.keys())[0]
    ingles_original = traducoes[primeira_chave]["ingles_original"].tolist()
    pares_modelos_prompts = list(combinations(traducoes.keys(), 2))

    # if LIMITE_FRASES:
    #     ingles_original = ingles_original[:LIMITE_FRASES]
    num_frases = len(ingles_original)

    from bert_score import BERTScorer
    scorer = BERTScorer(lang="pt", device="cuda", batch_size=64)

    saida_analise = os.path.join(BASE_DIR, f"dataset_{dataset_nome}", "[CSV] analise_modelos_prompts", f"analise_consistencia_{dataset_nome}.csv")
    saida_alertas = os.path.join(BASE_DIR, f"dataset_{dataset_nome}", "[CSV] analise_modelos_prompts", f"analise_discrepantes_{dataset_nome}.csv")
    saida_todas = os.path.join(BASE_DIR, f"dataset_{dataset_nome}", "[CSV] analise_modelos_prompts", f"todas_comparacoes_{dataset_nome}.csv")

    preparar_arquivo_saida(saida_analise)
    preparar_arquivo_saida(saida_todas)
    preparar_arquivo_saida(saida_alertas)

    # Calcular bertscore para todos os pares, organizando por frase
    print(f"\n{'='*60}")
    print(f"Analisando dataset: {dataset_nome}")
    print(f"Total de frases: {num_frases}")
    print(f"{'='*60}\n")
    
    resultados_buffer = []
    alertas_buffer = []
    todas_comparacoes_buffer = []
    num_consistentes = 0
    num_discrepantes = 0
    houve_alertas = False
    
    for inicio_lote in tqdm(range(0, num_frases, BATCH_SALVAMENTO), desc="Processando frases", unit="lote"):
        fim_lote = min(inicio_lote + BATCH_SALVAMENTO, num_frases)

        refs_lote = []
        cands_lote = []
        metadados_lote = []

        for idx in range(inicio_lote, fim_lote):
            for (modelo_a, prompt_a), (modelo_b, prompt_b) in pares_modelos_prompts:
                _a = str(traducoes[(modelo_a, prompt_a)].iloc[idx]["portugues_traduzido"]).strip()
                _b = str(traducoes[(modelo_b, prompt_b)].iloc[idx]["portugues_traduzido"]).strip()
                trad_a = _a if _a and _a.lower() != "nan" else "[vazio]"
                trad_b = _b if _b and _b.lower() != "nan" else "[vazio]"

                prompt_a_label = prompt_a if prompt_a is not None else "sem_prompt"
                prompt_b_label = prompt_b if prompt_b is not None else "sem_prompt"

                refs_lote.append(trad_a)
                cands_lote.append(trad_b)
                metadados_lote.append({
                    "indice": idx,
                    "ingles_original": ingles_original[idx],
                    "par": f"{modelo_a}/{prompt_a_label} X {modelo_b}/{prompt_b_label}",
                    "modelo_a": modelo_a,
                    "prompt_a": prompt_a_label,
                    "modelo_b": modelo_b,
                    "prompt_b": prompt_b_label,
                    "trad_a": trad_a,
                    "trad_b": trad_b,
                })

        if metadados_lote:
            _, _, f1_lote = scorer.score(cands_lote, refs_lote, verbose=False)
            scores_lote = f1_lote.tolist()
        else:
            scores_lote = []

        detalhes_por_frase = {idx: [] for idx in range(inicio_lote, fim_lote)}
        for meta, bertscore in zip(metadados_lote, scores_lote):
            detalhes_por_frase[meta["indice"]].append({
                "par": meta["par"],
                "modelo_a": meta["modelo_a"],
                "prompt_a": meta["prompt_a"],
                "modelo_b": meta["modelo_b"],
                "prompt_b": meta["prompt_b"],
                "trad_a": meta["trad_a"],
                "trad_b": meta["trad_b"],
                "bertscore": bertscore,
            })

        for idx in range(inicio_lote, fim_lote):
            detalhes_pares = detalhes_por_frase[idx]
            scores_frase = [p["bertscore"] for p in detalhes_pares]

            mediana = np.median(scores_frase)
            limite_inferior = mediana * (1 - VARIACAO_PERMITIDA)
            limite_superior = mediana * (1 + VARIACAO_PERMITIDA)

            pares_discrepantes = [
                p for p in detalhes_pares
                if p["bertscore"] < limite_inferior
            ]

            status = "consistente" if len(pares_discrepantes) == 0 else "discrepante"

            resultado_frase = {
                "indice": idx,
                "ingles_original": ingles_original[idx],
                "mediana_bertscore": round(mediana, 2),
                "limite_inferior": round(limite_inferior, 2),
                "limite_superior": round(limite_superior, 2),
                "num_pares_total": len(scores_frase),
                "num_pares_discrepantes": len(pares_discrepantes),
                "status": status,
                "status_manual": "",
            }
            resultados_buffer.append(resultado_frase)

            if status == "consistente":
                num_consistentes += 1
            else:
                num_discrepantes += 1

            for par in detalhes_pares:
                par_status = "discrepante" if par["bertscore"] < limite_inferior else "consistente"
                todas_comparacoes_buffer.append({
                    "indice": idx,
                    "ingles_original": ingles_original[idx],
                    "modelo_a": par["modelo_a"],
                    "prompt_a": par["prompt_a"],
                    "trad_a": par["trad_a"],
                    "modelo_b": par["modelo_b"],
                    "prompt_b": par["prompt_b"],
                    "trad_b": par["trad_b"],
                    "bertscore": round(par["bertscore"], 2),
                    "mediana_bertscore": round(mediana, 2),
                    "limite_inferior": round(limite_inferior, 2),
                    "limite_superior": round(limite_superior, 2),
                    "status": par_status,
                })

            if pares_discrepantes:
                for par in pares_discrepantes:
                    alertas_buffer.append({
                        "indice": idx,
                        "ingles_original": ingles_original[idx],
                        "par_discrepante": par["par"],
                        "modelo_a": par["modelo_a"],
                        "prompt_a": par["prompt_a"],
                        "trad_a": par["trad_a"],
                        "modelo_b": par["modelo_b"],
                        "prompt_b": par["prompt_b"],
                        "trad_b": par["trad_b"],
                        "bertscore": round(par["bertscore"], 2),
                        "mediana_bertscore": round(mediana, 2),
                        "limite_inferior": round(limite_inferior, 2),
                        "limite_superior": round(limite_superior, 2),
                    })

        append_registros_csv(saida_analise, resultados_buffer)
        append_registros_csv(saida_todas, todas_comparacoes_buffer)
        if alertas_buffer:
            append_registros_csv(saida_alertas, alertas_buffer)
            houve_alertas = True

        resultados_buffer = []
        todas_comparacoes_buffer = []
        alertas_buffer = []
    
    # Salvar o restante que nao completou um batch
    append_registros_csv(saida_analise, resultados_buffer)
    append_registros_csv(saida_todas, todas_comparacoes_buffer)
    if alertas_buffer:
        append_registros_csv(saida_alertas, alertas_buffer)
        houve_alertas = True
    
    # Exibir menu
    print(f"Resultados salvos:")
    print(f"  - Analise: {saida_analise}")
    print(f"  - Todas as comparacoes: {saida_todas}")
    if houve_alertas:
        print(f"  - Alertas: {saida_alertas}")
    print(f"\nResumo:")
    print(f"  Frases consistentes: {num_consistentes} ({num_consistentes/num_frases*100:.1f}%)")
    print(f"  Frases discrepantes: {num_discrepantes} ({num_discrepantes/num_frases*100:.1f}%)")


def calcular_ranking(dataset_nome):
    arquivo_analise = os.path.join(BASE_DIR, f"dataset_{dataset_nome}", "[CSV] analise_modelos_prompts", f"analise_consistencia_{dataset_nome}.csv")
    
    if not os.path.exists(arquivo_analise):
        print(f"Erro: arquivo {arquivo_analise} nao encontrado.")
        print(f"Execute primeiro a analise de consistencia.")
        return
    
    df = pd.read_csv(arquivo_analise)
    
    # Usar status_manual se preenchido, senão usar status automático
    df["status_final"] = df.apply(
        lambda row: row["status_manual"] if pd.notna(row["status_manual"]) and row["status_manual"].strip() != "" 
        else row["status"], 
        axis=1
    )
    
    # Carregar traduções para mapear cada frase aos modelos
    base_dir = DATASETS[dataset_nome]
    traducoes = {}
    for modelo in MODELOS:
        for prompt in PROMPTS_POR_MODELO.get(modelo, []):
            try:
                df_trad = carregar_traducoes(base_dir, modelo, prompt)
                traducoes[(modelo, prompt)] = df_trad
            except FileNotFoundError:
                pass
    
    # Carregar arquivo de alertas para identificar pares discrepantes
    arquivo_alertas = os.path.join(BASE_DIR, f"dataset_{dataset_nome}", "[CSV] analise_modelos_prompts", f"analise_discrepantes_{dataset_nome}.csv")
    
    # Criar conjunto de modelos/prompts
    modelos_prompts = set()
    for modelo, prompt in traducoes.keys():
        prompt_label = prompt if prompt is not None else "sem_prompt"
        modelos_prompts.add(f"{modelo}/{prompt_label}")
    
    # Inicializar contadores: total de comparações e comparações consistentes
    stats = {mp: {"total_comparacoes": 0, "comparacoes_consistentes": 0} for mp in modelos_prompts}
    
    # Processar cada frase
    for idx, frase in df.iterrows():
        indice_frase = frase["indice"]
        
        # Para cada modelo/prompt, ele participa de (n-1) comparações por frase
        # onde n é o total de modelos/prompts
        num_outros = len(modelos_prompts) - 1
        
        for mp in modelos_prompts:
            stats[mp]["total_comparacoes"] += num_outros
            stats[mp]["comparacoes_consistentes"] += num_outros
    
    # Se houver alertas, subtrair os pares discrepantes
    if os.path.exists(arquivo_alertas):
        df_alertas = pd.read_csv(arquivo_alertas)
        
        for _, alerta in df_alertas.iterrows():
            modelo_a = alerta["modelo_a"]
            prompt_a = alerta["prompt_a"]
            modelo_b = alerta["modelo_b"]
            prompt_b = alerta["prompt_b"]
            
            mp_a = f"{modelo_a}/{prompt_a}"
            mp_b = f"{modelo_b}/{prompt_b}"
            
            # Ambos os lados do par discrepante perdem 1 ponto
            stats[mp_a]["comparacoes_consistentes"] -= 1
            stats[mp_b]["comparacoes_consistentes"] -= 1
    
    # Criar DataFrame de ranking
    ranking_data = []
    for mp, st in stats.items():
        percentual = (st["comparacoes_consistentes"] / st["total_comparacoes"] * 100) if st["total_comparacoes"] > 0 else 0
        ranking_data.append({
            "modelo_prompt": mp,
            "comparacoes_consistentes": st["comparacoes_consistentes"],
            "total_comparacoes": st["total_comparacoes"],
            "percentual_consistencia": round(percentual, 2)
        })
    
    ranking = pd.DataFrame(ranking_data).sort_values(["percentual_consistencia", "modelo_prompt"], ascending=[False, True])
    
    # Salvar ranking
    saida_ranking = os.path.join(BASE_DIR, f"dataset_{dataset_nome}", "[CSV] analise_modelos_prompts", f"ranking_consistencia_{dataset_nome}.csv")
    ranking.to_csv(saida_ranking, index=False)
    
    print(ranking.to_string(index=False))
    print(f"\nRanking salvo em: dataset_{dataset_nome}/[CSV] analise_modelos_prompts/ranking_consistencia_{dataset_nome}.csv\n")


def calculo_rouge(rouge_inst, ref, hyp):
    try:
        scores = rouge_inst.get_scores(hyp, ref)[0]
        return {
            "rouge1": round(scores["rouge-1"]["f"], 4),
            "rouge2": round(scores["rouge-2"]["f"], 4),
            "rougeL": round(scores["rouge-l"]["f"], 4),
        }
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def gerar_frases_consistentes(dataset_nome):
    
    arquivo_todas = os.path.join(
        BASE_DIR, f"dataset_{dataset_nome}",
        "[CSV] analise_modelos_prompts",
        f"todas_comparacoes_{dataset_nome}.csv",
    )

    if not os.path.exists(arquivo_todas):
        print(f"Erro: arquivo nao encontrado: {arquivo_todas}")
        print("Execute primeiro a analise de consistencia.")
        return

    df = pd.read_csv(arquivo_todas)

    # Respeitar status_manual se preenchido
    if "status_manual" in df.columns:
        df["status_final"] = df.apply(
            lambda row: row["status_manual"]
            if pd.notna(row["status_manual"]) and str(row["status_manual"]).strip() != ""
            else row["status"],
            axis=1,
        )
    else:
        df["status_final"] = df["status"]

    consistentes = df[df["status_final"] == "consistente"].copy()

    if consistentes.empty:
        print(f"Nenhum par consistente encontrado em {dataset_nome}.")
        return

    scorer_rouge = RougeLib()

    resultados = []
    for _, row in tqdm(consistentes.iterrows(), total=len(consistentes), desc="Calculando ROUGE", unit="par"):
        rouge = calculo_rouge(scorer_rouge, str(row["trad_a"]), str(row["trad_b"]))
        resultados.append({
            "indice": row["indice"],
            "ingles_original": row["ingles_original"],
            "modelo_a": row["modelo_a"],
            "prompt_a": row["prompt_a"],
            "trad_a": row["trad_a"],
            "modelo_b": row["modelo_b"],
            "prompt_b": row["prompt_b"],
            "trad_b": row["trad_b"],
            "bertscore": row["bertscore"],
            "mediana_bertscore": row["mediana_bertscore"],
            "limite_inferior": row["limite_inferior"],
            "limite_superior": row["limite_superior"],
            "status": row["status"],
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
        })

    saida = os.path.join(
        BASE_DIR, f"dataset_{dataset_nome}",
        "[CSV] analise_modelos_prompts",
        f"frases_consistentes_{dataset_nome}.csv",
    )
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    pd.DataFrame(resultados).to_csv(saida, index=False)

    print(f"Frases consistentes com ROUGE salvas em: {saida}")
    print(f"  Total de pares consistentes: {len(resultados)}")


if __name__ == "__main__":
    # Verificar modo de execução
    if len(sys.argv) > 1 and sys.argv[1] == "--calcular-ranking":
        # calcular ranking após ajustes manuais
        if len(sys.argv) < 3:
            print("Datasets disponiveis: newsmet, manual_data")
            sys.exit(1)
        
        dataset_nome = sys.argv[2]
        if dataset_nome not in DATASETS:
            print(f"Erro: dataset '{dataset_nome}' nao encontrado.")
            print(f"Datasets disponiveis: {', '.join(DATASETS.keys())}")
            sys.exit(1)
        
        calcular_ranking(dataset_nome)
    elif len(sys.argv) > 1 and sys.argv[1] == "--frases-consistentes":
        if len(sys.argv) < 3:
            print("Datasets disponiveis: newsmet, manual_data")
            sys.exit(1)
        dataset_nome = sys.argv[2]
        if dataset_nome not in DATASETS:
            print(f"Erro: dataset '{dataset_nome}' nao encontrado.")
            print(f"Datasets disponiveis: {', '.join(DATASETS.keys())}")
            sys.exit(1)
        gerar_frases_consistentes(dataset_nome)
    else:
        # análise de consistência inicial
        for dataset_nome, base_dir in DATASETS.items():
            analisar_consistencia(dataset_nome, base_dir)



