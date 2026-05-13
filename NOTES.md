# Metaphor - Back Translation

Este projeto visa estudar como LLMs fazem o back translation que consiste traduzir uma frase de uma língua A para B e novamente traduzir da língua B para A. No nosso contexto, estamos usando datasets do huggingface que contém frases em inglês com ou sem metáforas. 

Além disso, iremos usar algumas métricas para julgar se a tradução está sendo boa ou não, usaremos [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge), [BLEU](https://huggingface.co/spaces/evaluate-metric/bleu), [BERTScore](https://huggingface.co/spaces/evaluate-metric/bertscore), [BLEURT](https://huggingface.co/spaces/evaluate-metric/bleurt) e [COMET](https://huggingface.co/spaces/evaluate-metric/comet), todas estão documentadas no huggingface.

Nessa pasta, ocorre o desenvolvimento da terceira vertente do projeto que consiste em fazer BackTranslation. Aqui iremos escolher dois datasets de metáforas em inglês para realizar a tradução dele para o português pelas LLMs e novamente para o inglês para checarmos se as frases se mantêm iguais. Para o dataset, escolhemos os menores que temos separados, que no caso foram os arquivos `manual_data.parquet` com 718 frases e o `newsmet.csv` com 2592 frases.

Dito isso, devemos rodar as frases do inglês para o português e depois do português para o inglês novamente. 

No arquivo `genENtoPT` temos:
```json
{
    "fraseEN": "abc",
    "traducaoPT": "dfg"
}
```

Depois, pegamos esse "traducaoPT" para virar a nova frase no arquivo `genPTtoEN`
```json
{
    "frasePT": "dfg",
    "traducaoEN": "hij"
}
```


Ordem que estou rodando os scripts:
1) `genENtoPT_*.py`
2) `genPTtoEN_*.py`
3) `scripts/arquivos_dados/juntar_jsons.py`
4) `scripts/arquivos_dados/metricas.py`
5) `scripts/arquivos_dados/rodar_comet.py`
6) `scripts/tokens/tokens*.py`
7) `scripts/selecionar_frases/convert_xlxs.py` (opcional, para exportar para Excel)

\* Existem algumas variações no nome, várias opções de arquivos com a mesma sintaxe.


Na pasta `/comparacao_datasets` ficam os datasets.

Na pasta `/generate` tem os arquivos Python pra rodar tanto EN->PT quanto PT->EN
- `*_gemini.py` (Pro Gemini)
- `*_gemmaX.py` (Pro GemmaX)
- `*_gpt.py` (Pro Gpt)
- `*_locais.py` (Pros modelos gerais do huggingface)
- `trad*.py` (Pros modelos específicos de tradução do huggingface)

Na pasta script tem os códigos Python para diversas tarefas, geralmente um pra cada análise que fizemos em cima das traduções

Para ter os tokens, rodar os códigos `script/tokens*.py`

Os outros scripts foram pra outras tarefas muito específicas, cálculo de ranking de frases, seleção das melhores e piores, cálculo de media e desvio padrão, além de aplicação do ICC. Se precisar de algum desses pode me falar que explícito a ordem certinha de uso 

## Escolha de melhores prompts e modelos

Chegamos numa fase da pesquisa que depois de muitas traduções queremos analisar os melhores modelos e prompts para gerarmos nosso dataset de metáforas em português e dar continuidade na pesquisa.

Os scripts `analise_quartis_modelos` e `analise_quartis_prompts_modelos` leem a planilha de anotações manuais realizada pela Isabella que julga as melhores e piores traduções de cada frase, nos ajudando a rankear os pares modelo/prompt que forem melhores. Esses scripts geram gráficos que nós analisamos com muito cuidado até selecionarmos os modelos e pompts que seguiremos trabalhando em cima

* [Modelos escolhidos](#modelos-usados): Gpt, Gemini, GemmaX e Gemma3
* [Prompts escolhidos](#prompts-usados): Prompt 2 e Prompt 4


## Links dos datasets utilizados:

- https://huggingface.co/datasets/Sasidhar1826/manual_data_on_metaphors

- https://github.com/AxleBlaze3/NewsMet_Metaphor_Dataset


## Modelos usados:

- gpt-4o-mini
- gemini-2.0-flash-lite
- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct
- nllb-200-3.3B
- Ministral-8B-Instruct-2410
- opus-mt-en-ROMANCE (tem que forçar o pt na frase pq ele mistura mt espanhol)
- google/gemma-3-12b-it
- ModelSpace/GemmaX2-28-9B-v0.1

## Prompts usados:

- **Prompt 1:** "Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso"
- **Prompt 2:** "Traduzir a frase '{frase}' do inglês para o português. Apenas escreva a frase traduzida, nada além disso. A frase pode ou não conter metáfora"
- **Prompt 3:** "Você é um especialista em metáforas e tradução criativa. Traduza {frase} para o português, mantendo o sentido metafórico original. Responda apenas com a tradução."
- **Prompt 4:** "Você é um especialista em metáforas e tradução criativa. Somente traduza {frase} para o português, mantendo o sentido metafórico original. Por exemplo, 'kick the bucket' deve ser traduzido como 'bater as botas', e não como 'chutar o balde'. Responda apenas com a tradução."

***OBS:*** Não rodamos os prompts nos modelos nllb, opus e gemmaX porque eles não usam prompt, como são puramente de tradução, só recebem a frase como input.


## Scripts por pasta

**scripts/analise/**
- analise_quartis_modelos.py
- analise_quartis_prompts_modelos.py
- comparacao_modelos_prompt.py
- icc.py
- media_desvio_junto.py
- media_desvio_separado.py

**scripts/arquivos_dados/**
- juntar_jsons.py
- matriz.py
- metricas.py
- rodar_comet.py
- tamanho_matrizes.py

**scripts/metricas/**
- anotacao_manual_metricas.py
- melhor_traducao_xcomet.py

**scripts/selecionar_frases/**
- anotacao_frases_quartis.py
- convert_xlxs.py
- selecionando_frases.py
- selecionando_frases_quartis.py

**scripts/tokens/**
- tokens.py
- tokens_gemini.py
- tokens_gemma3.py
- tokens_gemmaX.py
- tokens_gpt.py

## O que cada script faz?

### Arquivos para analisar os dados:

**scripts/analise/analise_quartis_modelos.py**
Realiza análise estatística dos resultados dos modelos, separando e avaliando as traduções por quartis de desempenho, permitindo identificar padrões de qualidade entre diferentes modelos.

**scripts/analise/analise_quartis_prompts_modelos.py**
Faz análise dos resultados considerando tanto modelos quanto prompts, separando as traduções em quartis para avaliar o impacto dos diferentes prompts na qualidade das traduções, gerando gráficos que nos permitiram escolher quais modelos e pormpts continuar analisando.

**scripts/analise/comparacao_modelos_prompt.py**
Compara traduções de diferentes modelos e prompts, analisando consistência, recalculando rankings e gerando relatórios e gráficos comparativos de desempenho.

**scripts/analise/icc.py**
Calcula o coeficiente de correlação intraclasse (ICC) para avaliar a consistência entre diferentes avaliadores ou métricas nas anotações manuais das traduções.

**scripts/analise/media_desvio_junto.py**
Calcula médias e desvios padrão das métricas de tradução para todos os modelos juntos, permitindo uma visão geral do desempenho agregado.

**scripts/analise/media_desvio_separado.py**
Calcula médias e desvios padrão das métricas de tradução separadamente para cada modelo, facilitando a comparação individual de desempenho.

### Arquivos para organizar os dados dos datasets:

**scripts/arquivos_dados/juntar_jsons.py**
Une os arquivos ENtoPT.json e PTtoEN.json de cada modelo/prompt/dataset, gerando um único frases_traduzidas.json com as frases originais, traduzidas para o português e novamente para o inglês, facilitando o processamento posterior.

**scripts/arquivos_dados/matriz.py**
Gera o arquivo matriz.csv para cada modelo/prompt/dataset, já incluindo frases originais/traduzidas, label, rankings das métricas e a coluna Soma_ranking, padronizando o formato para etapas posteriores.

**scripts/arquivos_dados/metricas.py**
Calcula automaticamente as métricas ROUGE, BLEU, BERTScore e BLEURT para cada frase traduzida, salvando os resultados em frases_traduzidas_com_metricas.json para cada modelo/prompt/dataset.

**scripts/arquivos_dados/rodar_comet.py**
Executa modelos de avaliação automática (COMET22, KIWI-XL, XCOMET-XL) sobre as traduções geradas, salvando os scores dessas métricas para cada frase.

**scripts/arquivos_dados/tamanho_matrizes.py**
Verifica e imprime o tamanho (linhas e colunas) de cada arquivo matriz.csv, ajudando a identificar inconsistências ou problemas nos dados gerados.

### Arquivos para analisar as métricas:

**scripts/metricas/anotacao_manual_metricas.py**
Calcula métricas automáticas e estatísticas a partir das anotações manuais feitas sobre as traduções, permitindo comparar avaliações humanas e automáticas.

**scripts/metricas/criando_dataset_score_combinado.py**
Seleciona, para cada frase, a melhor tradução com base no score combinado entre COMET22, XCOMET-XL e KIWI-XL, separando os resultados por quartis e por tipo de frase (metafórica ou literal) para criar o dataset final.

### Arquivos para selecionar frases para análise manual:

**scripts/selecionar_frases/anotacao_frases_quartis.py**
Auxilia na anotação manual de frases selecionadas por quartis, facilitando a avaliação humana de subconjuntos representativos das traduções geradas pelos modelos.

**scripts/selecionar_frases/convert_xlxs.py**
Converte arquivos frases_traduzidas.json em planilhas Excel (.xlsx) para facilitar a visualização e análise manual dos dados.

**scripts/selecionar_frases/selecionando_frases_quartis.py**
Seleciona frases de acordo com os quartis das métricas, permitindo focar em subconjuntos de maior ou menor qualidade para análise ou anotação.

**scripts/selecionar_frases/selecionando_frases.py**
Seleciona frases específicas do conjunto de traduções com base em critérios definidos (como melhores ou piores rankings), para análises ou anotações posteriores.

### Arquivos para tokenizar as frases:

**scripts/tokens/tokens.py, scripts/tokens/tokens_gemini.py, scripts/tokens/tokens_gemma3.py, scripts/tokens/tokens_gemmaX.py, scripts/tokens/tokens_gpt.py**
Calculam e salvam a contagem de tokens das frases originais e traduzidas para diferentes modelos, permitindo análises de custo e complexidade das traduções.
