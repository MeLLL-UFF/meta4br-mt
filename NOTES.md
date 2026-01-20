# Metaphor - Back Translation

Este projeto visa estudar como LLMs (gpt, qwen, mistral e llama) fazem o back translation que consiste traduzir uma frase de uma língua A para B e novamente traduzir da língua B para A. No nosso contexto, estamos usando o dataset [common.parquet](https://huggingface.co/datasets/Sasidhar1826/common_metaphors_detection_dataset) do huggingface que contém 600 frases em inglês com ou sem metáforas. 

Além disso, iremos usar algumas métricas para julgar se a tradução está sendo boa ou não, usaremos [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge), [BLEU](https://huggingface.co/spaces/evaluate-metric/bleu), [BERTScore](https://huggingface.co/spaces/evaluate-metric/bertscore), [BLEURT](https://huggingface.co/spaces/evaluate-metric/bleurt) e [COMET](https://huggingface.co/spaces/evaluate-metric/comet), todas estão documentadas no huggingface.

Nessa pasta, ocorre o desenvolvimento da terceira vertente do projeto que consiste em fazer BackTranslation. Aqui iremos escolher dois datasets de metáforas em inglês para realizar a tradução dele para o português pelas LLMs e novamente para o inglês para checarmos se as frases se mantêm iguais. Para o dataset, escolhemos os menores que temos separados, que no caso foram os arquivos "common.parquet" com 70 frases e o "manual_data.parquet" com 718 frases.

Dito isso, devemos rodar as frases do inglês para o português e depois do português para o inglês novamente. 

No arquivo genENtoPT temos:
{
    "fraseEN: "abc",
    "traducaoPT": "dfg"
}

Depois, pegamos esse "traducaoPT" para virar a nova frase no arquivo genPTtoEN
{
    "frasePT": "dfg",
    "traducaoEN": "hij"
}

Ordem que estou rodando os scripts:
1) genENtoPT_*.py
2) genPTtoEN_*.py
3) script/juntar_jsons.py
4) script/metricas.py
5) script/rodar_comet.py
6) script/tokens*.py

* Indica que existem algumas variações no nome, várias opções de arquivos com a mesma sintaxe.


Na pasta comparacao_datasets ficam os datasets

Na pasta generate tem os arquivos Python pra rodar tanto EN->PT quanto PT->EN
- *_gemini.py (Pro Gemini)
- *_gemmaX.py (Pro GemmaX)
- *_gpt.py (Pro Gpt)
- *_locais.py (Pros modelos gerais do huggingface)
- trad*.py (Pros modelos específicos de tradução do huggingface)

Na pasta script tem os códigos Python para diversas tarefas, geralmente um pra cada análise que fizemos em cima das traduções

Para ter os tokens, rodar os códigos script/tokens*.py

Os outros scripts foram pra outras tarefas muito específicas, cálculo de ranking de frases, seleção das melhores e piores, cálculo de media e desvio padrão, além de aplicação do ICC. Se precisar de algum desses pode me falar que explícito a ordem certinha de uso 


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
- opus-mt-ROMANCE-en

Criamos um prompt2 com mais detalhes, mas não rodamos ele nos modelos da meta nem no marian porque eles não usam prompt, como sào puramente de tradução, só recebem a frase como input.

Na pasta 'analise_quartis' tem análise dos modelos e prompts, para ajudar a escolher um modelo e prompt para dar continuidade no projeto fazendo um adapter.