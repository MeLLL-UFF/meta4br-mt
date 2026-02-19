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
3) `script/juntar_jsons.py`
4) `script/metricas.py`
5) `script/rodar_comet.py`
6) `script/tokens*.py`

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

