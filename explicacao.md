Nessa pasta, ocorre o desenvolvimento da terceira vertente do projeto que consiste em fazer BackTranslation. Aqui iremos escolher um dataset de metáforas em inglês para realizar a tradução dele para o português pelas LLMs e novamente para o inglês para checarmos se as frases se mantêm iguais. Para o dataset, escolhemos o menor dos que temos separados, que no caso foi o arquivo "common.parquet" com 600 frases.

Dito isso, devemos rodar as farses do inglês para o português e depois do português para o inglês novamente. Para isso, iremos usar os seguintes prompts para conversar com as LLMs:


promptPT = **"Traduzir a frase <frase> do inglês para o português"** 

promptEN = **"Traduzir a frase <frase> do português para o inglês"**

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