from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def adicionar_labels(ax, fmt="{:.0f}", offset=0.5):
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

def separar_por_modelo_prompt(dataframe):
    """
    Separa por modelo_prompt completo (incluindo o prompt).
    Cada modelo/prompt é tratado como uma unidade separada.
    """
    dfs_por_modelo_prompt = {}
    
    for idx, row in dataframe.iterrows():
        modelos_prompt = str(row['modelo_prompt'])
        
        modelos_lista = [m.strip() for m in modelos_prompt.split(';') if m.strip()]
        
        for modelo_completo in modelos_lista:
            if modelo_completo not in dfs_por_modelo_prompt:
                dfs_por_modelo_prompt[modelo_completo] = []
            dfs_por_modelo_prompt[modelo_completo].append(row)
    
    for modelo_prompt in dfs_por_modelo_prompt:
        dfs_por_modelo_prompt[modelo_prompt] = pd.DataFrame(dfs_por_modelo_prompt[modelo_prompt])
    
    return dfs_por_modelo_prompt

def grafico_preservacao_metaforas(dfs_por_modelo_prompt, dataset_name):
    """
    Analisa a presença de metáfora na frase original e na traduzida, para analisar a preservação
    por modelo e prompt específicos
    """
    
    resultados = []
    
    for modelo_prompt, df in dfs_por_modelo_prompt.items():
        total = len(df)
        
        meta_orig_col = df['Contém metáfora na original?'].astype(str).str.lower()
        meta_trad_col = df['Contém metáfora na tradução?'].astype(str).str.lower()
        metafora_original = meta_orig_col.value_counts().get('sim', 0)
        metafora_traducao = meta_trad_col.value_counts().get('sim', 0)
        
        df_temp = df.copy()
        df_temp['original_sim'] = meta_orig_col == 'sim'
        df_temp['traducao_sim'] = meta_trad_col == 'sim'
        
        manteve_com_metafora = (df_temp['original_sim'] & df_temp['traducao_sim']).sum()
        perdeu_metafora = (df_temp['original_sim'] & ~df_temp['traducao_sim']).sum()
        ganhou_metafora = (~df_temp['original_sim'] & df_temp['traducao_sim']).sum()
        manteve_sem_metafora = (~df_temp['original_sim'] & ~df_temp['traducao_sim']).sum()
        
        resultados.append({
            'modelo_prompt': modelo_prompt,
            'total': total,
            'metafora_original': metafora_original,
            'metafora_traducao': metafora_traducao,
            'perdeu_metafora': perdeu_metafora,
            'ganhou_metafora': ganhou_metafora,
            'manteve_sem_metafora': manteve_sem_metafora,
            'manteve_com_metafora': manteve_com_metafora,
            'taxa_preservacao': (manteve_com_metafora / metafora_original * 100) if metafora_original > 0 else 0
        })
    
    df_resultados = pd.DataFrame(resultados).sort_values('total', ascending=False)
    
    # Gráfico 1: Comparação Original vs Tradução
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Subplot 1: Metáforas Original vs Tradução
    ax1 = axes[0, 0]
    x = range(len(df_resultados))
    width = 0.35
    cor_original = '#c084fc'  
    cor_traducao = '#60a5fa'  
    ax1.bar([i - width/2 for i in x], df_resultados['metafora_original'], width, label='Original', color=cor_original, alpha=0.85)
    ax1.bar([i + width/2 for i in x], df_resultados['metafora_traducao'], width, label='Tradução', color=cor_traducao, alpha=0.85)
    ax1.set_xlabel('Modelo/Prompt')
    ax1.set_ylabel('Quantidade de Metáforas')
    ax1.set_title('Metáforas: Original vs Tradução')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_resultados['modelo_prompt'], rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    adicionar_labels(ax1)
    
    # Subplot 2: Preservadas, Perdidas e alteradas
    ax2 = axes[0, 1]
    x = range(len(df_resultados))
    width = 0.2
    cores_categorias = {
        'manteve_com_metafora': '#ec4899',  
        'perdeu_metafora': '#f97316',       
        'ganhou_metafora': '#8b5cf6',       
        'manteve_sem_metafora': '#67e8f9',  
    }
    ax2.bar([i - 1.5 * width for i in x], df_resultados['manteve_com_metafora'], width, label='Manteve metáfora (sim -> sim)', color=cores_categorias['manteve_com_metafora'], alpha=0.9)
    ax2.bar([i - 0.5 * width for i in x], df_resultados['perdeu_metafora'], width, label='Perdeu metáfora (sim -> não)', color=cores_categorias['perdeu_metafora'], alpha=0.9)
    ax2.bar([i + 0.5 * width for i in x], df_resultados['ganhou_metafora'], width, label='Ganhou metáfora (não -> sim)', color=cores_categorias['ganhou_metafora'], alpha=0.9)
    ax2.bar([i + 1.5 * width for i in x], df_resultados['manteve_sem_metafora'], width, label='Manteve sem metáfora (não -> não)', color=cores_categorias['manteve_sem_metafora'], alpha=0.9)
    ax2.set_xlabel('Modelo/Prompt')
    ax2.set_ylabel('Quantidade')
    ax2.set_title('Análise de Preservação de Metáforas')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_resultados['modelo_prompt'], rotation=45, ha='right', fontsize=9)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    adicionar_labels(ax2)
    
    # Subplot 3: Taxa de Preservação (%)
    ax3 = axes[1, 0]
    colors = ['#ec4899' if x >= 80 else '#c084fc' if x >= 60 else '#60a5fa' for x in df_resultados['taxa_preservacao']]
    ax3.bar(range(len(df_resultados)), df_resultados['taxa_preservacao'], color=colors, alpha=0.9)
    ax3.set_xlabel('Modelo/Prompt')
    ax3.set_ylabel('Taxa de Preservação (%)')
    ax3.set_title('Taxa de Preservação de Metáforas por Modelo/Prompt')
    ax3.set_xticks(range(len(df_resultados)))
    ax3.set_xticklabels(df_resultados['modelo_prompt'], rotation=45, ha='right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    adicionar_labels(ax3, fmt="{:.1f}%", offset=1.5)
    
    # Subplot 4: Tabela de resumo
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    tabela_dados = df_resultados[['modelo_prompt', 'total', 'metafora_original', 'metafora_traducao', 'perdeu_metafora', 'ganhou_metafora', 'manteve_sem_metafora', 'manteve_com_metafora', 'taxa_preservacao']].copy()
    tabela_dados['taxa_preservacao'] = tabela_dados['taxa_preservacao'].round(1).astype(str) + '%'
    tabela_dados.columns = ['Modelo/Prompt', 'Total', 'Original', 'Tradução', 'Perdeu', 'Ganhou', 'Sem->Sem', 'Com->Com', 'Taxa (%)']
    
    table = ax4.table(cellText=tabela_dados.values, colLabels=tabela_dados.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    for i in range(len(tabela_dados.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()

    output_dir = Path("analise_quartis/prompts_modelos/preservacao_metaforas")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return df_resultados


def analisar_equivalencia(dfs_por_modelo_prompt, dataset_name):
    """Conta combinações de Equivalência Total/Parcial/Não-equivalência por modelo/prompt."""

    combos_ordem = [
        ('sim', 'sim', 'sim'),
        ('sim', 'sim', 'nao'),
        ('sim', 'nao', 'sim'),
        ('sim', 'nao', 'nao'),
        ('nao', 'sim', 'sim'),
        ('nao', 'sim', 'nao'),
        ('nao', 'nao', 'sim'),
        ('nao', 'nao', 'nao'),
    ]

    resultados = []
    heatmap_data = {}
    total_equivalencia = {'total': 0, 'parcial': 0, 'nao': 0}

    for modelo_prompt, df in dfs_por_modelo_prompt.items():
        df_temp = df.copy()
        df_temp['eq_total'] = df_temp['Equivalência Total'].astype(str).str.lower() == 'sim'
        df_temp['eq_parcial'] = df_temp['Equivalência Parcial'].astype(str).str.lower() == 'sim'
        df_temp['nao_eq'] = df_temp['Não-equivalência'].astype(str).str.lower() == 'sim'

        total_equivalencia['total'] += df_temp['eq_total'].sum()
        total_equivalencia['parcial'] += df_temp['eq_parcial'].sum()
        total_equivalencia['nao'] += df_temp['nao_eq'].sum()

        contagens = {}
        for combo in combos_ordem:
            mask = (
                (df_temp['eq_total'] == (combo[0] == 'sim')) &
                (df_temp['eq_parcial'] == (combo[1] == 'sim')) &
                (df_temp['nao_eq'] == (combo[2] == 'sim'))
            )
            contagens['/'.join(combo)] = mask.sum()

        heatmap_data[modelo_prompt] = contagens

        resultados.append({
            'modelo_prompt': modelo_prompt,
            'total': len(df_temp),
            **contagens
        })

    df_heatmap = pd.DataFrame(heatmap_data).reindex(["/".join(c) for c in combos_ordem])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    sns.heatmap(df_heatmap, annot=True, fmt='.0f', cmap='RdPu', cbar_kws={'label': 'Contagem'}, ax=ax1)
    ax1.set_title('Combinações de Equivalência por modelo/prompt')
    ax1.set_xlabel('Modelo/Prompt')
    ax1.set_ylabel('Combo (Total/Parcial/Não)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    cores_equiv = {'total': '#e91e63', 'parcial': '#9f0886', 'nao': "#5a0796"}
    tipos = ['total', 'parcial', 'nao']
    valores = [total_equivalencia[t] for t in tipos]
    rotulos = ['Total', 'Parcial', 'Não-equivalente']
    cores = [cores_equiv[t] for t in tipos]

    bars = ax2.bar(rotulos, valores, color=cores, alpha=0.85)
    ax2.set_ylabel('Quantidade')
    ax2.set_title('Total de Equivalências por Tipo')
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    output_dir = Path("analise_quartis/prompts_modelos/equivalencia")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resultados).sort_values('total', ascending=False).to_json(
        output_dir / f"{dataset_name}.json",
        orient='records', indent=2, force_ascii=False
    )


def metricas_numericas(dfs_por_modelo_prompt, dataset_name):
    """
    Analisa métricas numéricas de inteligibilidade e compreensão (escala 1-5)
    por modelo/prompt e gera gráficos com as médias e contagem de notas.
    """
    
    resultados_media = []
    resultados_contagem = []
    
    for modelo_prompt, df in dfs_por_modelo_prompt.items():
        # Converter para numérico, ignorando valores não conversíveis
        inteligibilidade = pd.to_numeric(df['Inteligibilidade'], errors='coerce')
        compreensao = pd.to_numeric(df['Compreensão'], errors='coerce')
        
        resultados_media.append({
            'modelo_prompt': modelo_prompt,
            'total': len(df),
            'inteligibilidade_media': inteligibilidade.mean(),
            'compreensao_media': compreensao.mean(),
        })
        
        # Contar cada nota de 1 a 5
        intel_counts = inteligibilidade.value_counts().sort_index()
        compr_counts = compreensao.value_counts().sort_index()
        
        # Garantir que todas as notas de 1-5 apareçam (mesmo que com 0)
        for nota in range(1, 6):
            if nota not in intel_counts.index:
                intel_counts[nota] = 0
            if nota not in compr_counts.index:
                compr_counts[nota] = 0
        
        intel_counts = intel_counts.sort_index()
        compr_counts = compr_counts.sort_index()
        
        resultados_contagem.append({
            'modelo_prompt': modelo_prompt,
            **{f'intel_{i}': int(intel_counts.get(i, 0)) for i in range(1, 6)},
            **{f'compr_{i}': int(compr_counts.get(i, 0)) for i in range(1, 6)},
        })
    
    df_media = pd.DataFrame(resultados_media).sort_values('inteligibilidade_media', ascending=False)
    df_contagem = pd.DataFrame(resultados_contagem)
    
    # Criar figura com 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    # Gráfico 1: Médias
    ax1 = axes[0]
    x = range(len(df_media))
    width = 0.35
    cor_inteligibilidade = '#c084fc'  # Lilás
    cor_compreensao = '#ec4899'      # Rosa
    
    ax1.bar([i - width/2 for i in x], df_media['inteligibilidade_media'], width, 
            label='Inteligibilidade', color=cor_inteligibilidade, alpha=0.85)
    ax1.bar([i + width/2 for i in x], df_media['compreensao_media'], width, 
            label='Compreensão', color=cor_compreensao, alpha=0.85)
    
    ax1.set_xlabel('Modelo/Prompt', fontsize=12)
    ax1.set_ylabel('Média de Pontuação', fontsize=12)
    ax1.set_title('Média: Inteligibilidade vs Compreensão', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_media['modelo_prompt'], rotation=45, ha='right', fontsize=9)
    ax1.set_ylim([0, 5])
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    adicionar_labels(ax1, fmt="{:.1f}", offset=0.5)
    
    # Gráfico 2: Contagem em Inteligibilidade
    ax2 = axes[1]
    x = range(len(df_contagem))
    width = 0.15
    notas = [1, 2, 3, 4, 5]
    cores_notas = ['#f8b4d6', '#ec4899', '#d946a6', '#c084fc', '#8b5cf6']
    
    for idx, nota in enumerate(notas):
        valores = [df_contagem[f'intel_{nota}'].iloc[i] for i in range(len(df_contagem))]
        ax2.bar([i + (idx - 2) * width for i in x], valores, width, 
                label=f'Nota {nota}', color=cores_notas[idx], alpha=0.85)
    
    ax2.set_xlabel('Modelo/Prompt', fontsize=12)
    ax2.set_ylabel('Quantidade', fontsize=12)
    ax2.set_title('Contagem de Notas em Inteligibilidade', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_contagem['modelo_prompt'], rotation=45, ha='right', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    adicionar_labels(ax2, fmt="{:.0f}", offset=0.3)
    
    # Gráfico 3: Contagem em Compreensão
    ax3 = axes[2]
    for idx, nota in enumerate(notas):
        valores = [df_contagem[f'compr_{nota}'].iloc[i] for i in range(len(df_contagem))]
        ax3.bar([i + (idx - 2) * width for i in x], valores, width, 
                label=f'Nota {nota}', color=cores_notas[idx], alpha=0.85)
    
    ax3.set_xlabel('Modelo/Prompt', fontsize=12)
    ax3.set_ylabel('Quantidade', fontsize=12)
    ax3.set_title('Contagem de Notas em Compreensão', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_contagem['modelo_prompt'], rotation=45, ha='right', fontsize=9)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    adicionar_labels(ax3, fmt="{:.0f}", offset=0.3)
    
    plt.tight_layout()
    
    output_dir = Path("analise_quartis/prompts_modelos/metricas_numericas")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Salvar resultados em JSON
    df_media.to_json(output_dir / f'{dataset_name}_metricas_media.json', 
                     orient='records', indent=2, force_ascii=False)
    df_contagem.to_json(output_dir / f'{dataset_name}_contagem_notas.json', 
                        orient='records', indent=2, force_ascii=False)
    
    return df_media, df_contagem


### MAIN ###

path = 'planilhas/Resposta - Análise dos quartis.xlsx'
mapeamento = {
    "manual_data": "Análise quartis - manual_data",
    "newsmet": "Análise quartis - newsmet",
}

Path("analise_quartis/prompts_modelos").mkdir(parents=True, exist_ok=True)

for dataset in ["manual_data", "newsmet"]:
    df = pd.read_excel(path, sheet_name=mapeamento[dataset])
    
    dfs_por_modelo_prompt = separar_por_modelo_prompt(df)
    resultados = grafico_preservacao_metaforas(dfs_por_modelo_prompt, dataset)
    
    output_dir = Path("analise_quartis/prompts_modelos/preservacao_metaforas")
    output_dir.mkdir(parents=True, exist_ok=True)
    resultados.to_json(output_dir / f'{dataset}.json', orient='records', indent=2, force_ascii=False)
    
    analisar_equivalencia(dfs_por_modelo_prompt, dataset)

# Executar análise de métricas numéricas
for dataset in ["manual_data", "newsmet"]:
    df = pd.read_excel(path, sheet_name=mapeamento[dataset])
    
    dfs_por_modelo_prompt = separar_por_modelo_prompt(df)
    metricas_numericas(dfs_por_modelo_prompt, dataset)
