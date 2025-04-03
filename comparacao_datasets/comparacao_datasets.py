import pandas as pd

# Substitua 'seu_arquivo.parquet' pelo caminho do seu arquivo .parquet
alsides = pd.read_parquet("BackTranslation/comparacao_datasets/alsides.parquet", engine='pyarrow') 
common = pd.read_parquet("BackTranslation/comparacao_datasets/common.parquet", engine='pyarrow') 
manual_data = pd.read_parquet("BackTranslation/comparacao_datasets/manual_data.parquet", engine='pyarrow') 
novel = pd.read_parquet("BackTranslation/comparacao_datasets/novel.parquet", engine='pyarrow') 
trofi = pd.read_parquet("BackTranslation/comparacao_datasets/trofi.parquet", engine='pyarrow') 
moh = pd.read_csv("BackTranslation/comparacao_datasets/moh.tsv", sep='\t')
metaphor_cot = pd.read_json("BackTranslation/comparacao_datasets/Metaphor_CoT_ft.json")



# Exibir a contagem de linhas do DataFrame
print("Número de linhas em alsides:", len(alsides), "\n")
print("Número de linhas em common :", len(common), "\n")
print("Número de linhas em manual_data:", len(manual_data), "\n")
print("Número de linhas em novel:", len(novel), "\n")
print("Número de linhas em trofi:", len(trofi), "\n")
print("Número de linhas em moh:", len(moh), "\n")
print("Número de linhas em metaphor_cot:", len(metaphor_cot), "\n\n")
print("Início do dataset common.parquet")

print(common.head())

