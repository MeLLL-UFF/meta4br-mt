# Metaphor - Back Translation

Este projeto visa estudar como LLMs (gpt, qwen, mistral e llama) fazem o back translation que consiste traduzir uma frase de uma língua A para B e novamente traduzir da língua B para A. No nosso contexto, estamos usando o dataset [common.parquet](https://huggingface.co/datasets/Sasidhar1826/common_metaphors_detection_dataset) do huggingface que contém 600 frases em inglês com ou sem metáforas. 

Além disso, iremos usar algumas métricas para julgar se a tradução está sendo boa ou não, usaremos [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge), [BLEU](https://huggingface.co/spaces/evaluate-metric/bleu), [BERTScore](https://huggingface.co/spaces/evaluate-metric/bertscore), [BLEURT](https://huggingface.co/spaces/evaluate-metric/bleurt) e [COMET](https://huggingface.co/spaces/evaluate-metric/comet), todas estão documentados no huggingface.

Nessa pasta, ocorre o desenvolvimento da terceira vertente do projeto que consiste em fazer BackTranslation. Aqui iremos escolher dois datasets de metáforas em inglês para realizar a tradução dele para o português pelas LLMs e novamente para o inglês para checarmos se as frases se mantêm iguais. Para o dataset, escolhemos os menores que temos separados, que no caso foram os arquivos "common.parquet" com 70 frases e o "manual_data.parquet" com 718 frases.

Dito isso, devemos rodar as farses do inglês para o português e depois do português para o inglês novamente. Para isso, iremos usar os seguintes prompts para conversar com as LLMs:


promptPT = **"Traduzir a frase <frase> do inglês para o português. Apenas escreva a frase traduzida, nada além disso"** 

promptEN = **"Traduzir a frase <frase> do português para o inglês. Apenas escreva a frase traduzida, nada além disso"**

Se analisarmos que as respostas foram muito confusas, podemos analisar a melhora desses prompts, incluindo maiores detalhes e contextos.

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
1) genENtoPT.py
2) genPTtoEN.py
3) script_juntar_jsons.py
4) script_metricas.py
5) script_comet.py
6) script_tokens.py

Citação:
**ROUGE**
- @inproceedings{lin-2004-rouge,
    title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",
    author = "Lin, Chin-Yew",
    booktitle = "Text Summarization Branches Out",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W04-1013",
    pages = "74--81",}

**BLEU**
- @INPROCEEDINGS{Papineni02bleu:a,
    author = {Kishore Papineni and Salim Roukos and Todd Ward and Wei-jing Zhu},
    title = {BLEU: a Method for Automatic Evaluation of Machine Translation},
    booktitle = {},
    year = {2002},
    pages = {311--318}
}
- @inproceedings{lin-och-2004-orange,
    title = "{ORANGE}: a Method for Evaluating Automatic Evaluation Metrics for Machine Translation",
    author = "Lin, Chin-Yew  and
      Och, Franz Josef",
    booktitle = "{COLING} 2004: Proceedings of the 20th International Conference on Computational Linguistics",
    month = "aug 23{--}aug 27",
    year = "2004",
    address = "Geneva, Switzerland",
    publisher = "COLING",
    url = "https://www.aclweb.org/anthology/C04-1072",
    pages = "501--507",
}

**BERTScore**
- @inproceedings{bert-score,
  title={BERTScore: Evaluating Text Generation with BERT},
  author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=SkeHuCVFDr}
}

**BLEURT**
- @inproceedings{bleurt,
  title={BLEURT: Learning Robust Metrics for Text Generation},
  author={Thibault Sellam and Dipanjan Das and Ankur P. Parikh},
  booktitle={ACL},
  year={2020},
  url={https://arxiv.org/abs/2004.04696}
}

**COMET**
- @inproceedings{rei-etal-2022-comet,
    title = "{COMET}-22: Unbabel-{IST} 2022 Submission for the Metrics Shared Task",
    author = "Rei, Ricardo  and
      C. de Souza, Jos{\'e} G.  and
      Alves, Duarte  and
      Zerva, Chrysoula  and
      Farinha, Ana C  and
      Glushkova, Taisiya  and
      Lavie, Alon  and
      Coheur, Luisa  and
      Martins, Andr{\'e} F. T.",
    booktitle = "Proceedings of the Seventh Conference on Machine Translation (WMT)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wmt-1.52",
    pages = "578--585",
}
- @inproceedings{rei-EtAl:2020:WMT,
   author    = {Rei, Ricardo  and  Stewart, Craig  and  Farinha, Ana C  and  Lavie, Alon},
   title     = {Unbabel's Participation in the WMT20 Metrics Shared Task},
   booktitle      = {Proceedings of the Fifth Conference on Machine Translation},
   month          = {November},
   year           = {2020},
   address        = {Online},
   publisher      = {Association for Computational Linguistics},
   pages     = {909--918},
}
- @inproceedings{rei-etal-2020-comet,
   title = "{COMET}: A Neural Framework for {MT} Evaluation",
   author = "Rei, Ricardo  and
      Stewart, Craig  and
      Farinha, Ana C  and
      Lavie, Alon",
   booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
   month = nov,
   year = "2020",
   address = "Online",
   publisher = "Association for Computational Linguistics",
   url = "https://www.aclweb.org/anthology/2020.emnlp-main.213",
   pages = "2685--2702",
}
- @inproceedings{rei-etal-2022-searching,
    title = "Searching for {COMETINHO}: The Little Metric That Could",
    author = "Rei, Ricardo  and
      Farinha, Ana C  and
      de Souza, Jos{\'e} G.C.  and
      Ramos, Pedro G.  and
      Martins, Andr{\'e} F.T.  and
      Coheur, Luisa  and
      Lavie, Alon",
    booktitle = "Proceedings of the 23rd Annual Conference of the European Association for Machine Translation",
    month = jun,
    year = "2022",
    address = "Ghent, Belgium",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2022.eamt-1.9",
    pages = "61--70",
}

https://huggingface.co/datasets/Sasidhar1826/manual_data_on_metaphors

Modelos usados:

- gpt-4o-mini
- gemini-2.0-flash-lite
- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct
- nllb-200-3.3B
- Ministral-8B-Instruct-2410
- opus-mt-en-ROMANCE (tem que forçar o pt na frase pq ele mistura mt espanhol)