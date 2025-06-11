import pandas as pd
import json

# Abrir e carregar o arquivo JSON
with open('dataset_common/gemmaX/frases_traduzidas.json', 'r', encoding='utf-8') as f:
    data = json.load(f)  # data é uma lista de dicionários

# Converter para DataFrame
df = pd.DataFrame(data)

# Salvar como Excel
df.to_excel('planilhas/gemmaX.xlsx', index=False)
