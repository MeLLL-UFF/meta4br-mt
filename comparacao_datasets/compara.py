import pandas as pd

# Substitua 'seu_arquivo.parquet' pelo caminho do seu arquivo .parquet
common1 = pd.read_parquet("comparacao_datasets/common1.parquet", engine='pyarrow') 
manual_data = pd.read_parquet("comparacao_datasets/manual_data.parquet", engine='pyarrow') 

df_common = pd.read_parquet("comparacao_datasets/common1.parquet")
df_manualdata= pd.read_parquet("comparacao_datasets/manual_data.parquet")

resp = []

if(len(df_common) < len(df_manualdata)):
    teste = True
else:
    teste = False

if(teste):
    for i in range(len(df_common)):
        if(df_common.iloc[i]['text'] in df_common['text'].values):
            resp.append(df_common.iloc[i]['text'])
else:
    for i in range(len(df_manualdata)):
        if(df_manualdata.iloc[i]['Sentence'] in df_manualdata['Sentence'].values):
            resp.append(df_manualdata.iloc[i]['Sentence'])

print(f"Dataset common sem duplicatas: {len(df_common)} frases")
print(f"Dataset manual_data sem duplicatas: {len(df_manualdata)} frases")
print(f"Temos {len(resp)} duplicatas entre os datasets")


