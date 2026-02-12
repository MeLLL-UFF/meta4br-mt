import os
from itertools import combinations
import numpy as np
import pandas as pd
from bert_score import score
import sys
from tqdm import tqdm

# Esses comandos serão usados caso altere a anotação das frases como consistente ou discrepante
# python scripts/comparacao_modelos_prompt.py --calcular-ranking newsmet
# python scripts/comparacao_modelos_prompt.py --calcular-ranking manual_data

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
    # "newsmet": os.path.join(BASE_DIR, "dataset_newsmet"),
    "manual_data": os.path.join(BASE_DIR, "dataset_manual_data"),
}
VARIACAO_PERMITIDA = 0.20  # Podemos mudar se necessário
LIMITE_FRASES = 20


def calcular_bertscore(frase_ref, frase_comparativa):
    P, R, f1 = score([frase_ref], [frase_comparativa], lang="pt", verbose=False)
    return f1.item()


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

    if LIMITE_FRASES:
        ingles_original = ingles_original[:LIMITE_FRASES]
    num_frases = len(ingles_original)

    # Calcular bertscore para todos os pares, organizando por frase
    print(f"\n{'='*60}")
    print(f"Analisando dataset: {dataset_nome}")
    print(f"Total de frases: {num_frases}")
    print(f"{'='*60}\n")
    
    resultados_por_frase = []
    alertas = []
    todas_comparacoes = []
    
    for idx in tqdm(range(num_frases), desc="Processando frases", unit="frase"):
        scores_frase = []
        detalhes_pares = []
        
        # Comparar todos os pares de modelos/prompts para esta frase
        for (modelo_a, prompt_a), (modelo_b, prompt_b) in combinations(traducoes.keys(), 2):
            trad_a = traducoes[(modelo_a, prompt_a)].iloc[idx]["portugues_traduzido"]
            trad_b = traducoes[(modelo_b, prompt_b)].iloc[idx]["portugues_traduzido"]
            bertscore = calcular_bertscore(trad_a, trad_b)
            
            prompt_a_label = prompt_a if prompt_a is not None else "sem_prompt"
            prompt_b_label = prompt_b if prompt_b is not None else "sem_prompt"
            
            scores_frase.append(bertscore)
            detalhes_pares.append({
                "par": f"{modelo_a}/{prompt_a_label} X {modelo_b}/{prompt_b_label}",
                "modelo_a": modelo_a,
                "prompt_a": prompt_a_label,
                "modelo_b": modelo_b,
                "prompt_b": prompt_b_label,
                "trad_a": trad_a,
                "trad_b": trad_b,
                "bertscore": bertscore
            })
        
        # Calcular mediana e limites
        mediana = np.median(scores_frase)
        limite_inferior = mediana * (1 - VARIACAO_PERMITIDA)
        limite_superior = mediana * (1 + VARIACAO_PERMITIDA)
        
        # Verificar quais pares estão fora do limite
        pares_discrepantes = [
            p for p in detalhes_pares 
            if p["bertscore"] < limite_inferior or p["bertscore"] > limite_superior
        ]
        
        # Determinar status da frase
        status = "consistente" if len(pares_discrepantes) == 0 else "discrepante"
        
        resultados_por_frase.append({
            "indice": idx,
            "ingles_original": ingles_original[idx],
            "mediana_bertscore": round(mediana, 2),
            "limite_inferior": round(limite_inferior, 2),
            "limite_superior": round(limite_superior, 2),
            "num_pares_total": len(scores_frase),
            "num_pares_discrepantes": len(pares_discrepantes),
            "status": status,
            "status_manual": "", 
        })
        
        # Adicionar todas as comparações ao csv completo
        for par in detalhes_pares:
            par_status = "discrepante" if par["bertscore"] < limite_inferior or par["bertscore"] > limite_superior else "consistente"
            todas_comparacoes.append({
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
        
        # Se houver discrepâncias
        if pares_discrepantes:
            for par in pares_discrepantes:
                alertas.append({
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
    
    # Salvar resultados
    df_analise = pd.DataFrame(resultados_por_frase)
    df_alertas = pd.DataFrame(alertas)
    df_todas_comparacoes = pd.DataFrame(todas_comparacoes)
    
    saida_analise = os.path.join(BASE_DIR, f"dataset_{dataset_nome}", "[CSV] analise_modelos_prompts", f"analise_consistencia_{dataset_nome}.csv")
    saida_alertas = os.path.join(BASE_DIR, f"dataset_{dataset_nome}", "[CSV] analise_modelos_prompts", f"analise_discrepantes_{dataset_nome}.csv")
    saida_todas = os.path.join(BASE_DIR, f"dataset_{dataset_nome}", "[CSV] analise_modelos_prompts", f"todas_comparacoes_{dataset_nome}.csv")
    
    df_analise.to_csv(saida_analise, index=False)
    df_todas_comparacoes.to_csv(saida_todas, index=False)
    if not df_alertas.empty:
        df_alertas.to_csv(saida_alertas, index=False)
    
    # Exibir menu
    num_consistentes = len(df_analise[df_analise["status"] == "consistente"])
    num_discrepantes = len(df_analise[df_analise["status"] == "discrepante"])
    
    print(f"Resultados salvos:")
    print(f"  - Analise: {saida_analise}")
    print(f"  - Todas as comparacoes: {saida_todas}")
    if not df_alertas.empty:
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
        
        # Todos os pares iniciam como consistentes
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
    else:
        # análise de consistência inicial
        for dataset_nome, base_dir in DATASETS.items():
            analisar_consistencia(dataset_nome, base_dir)



