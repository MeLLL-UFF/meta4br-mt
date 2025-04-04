import pandas as pd

# Substitua 'seu_arquivo.parquet' pelo caminho do seu arquivo .parquet
alsides = pd.read_parquet("comparacao_datasets/alsides.parquet", engine='pyarrow') 
common = pd.read_parquet("comparacao_datasets/common.parquet", engine='pyarrow')
common1 = pd.read_parquet("comparacao_datasets/common1.parquet", engine='pyarrow') 
manual_data = pd.read_parquet("comparacao_datasets/manual_data.parquet", engine='pyarrow') 
novel = pd.read_parquet("comparacao_datasets/novel.parquet", engine='pyarrow') 
trofi = pd.read_parquet("comparacao_datasets/trofi.parquet", engine='pyarrow') 
moh = pd.read_csv("comparacao_datasets/moh.tsv", sep='\t')
metaphor_cot = pd.read_json("comparacao_datasets/Metaphor_CoT_ft.json")



# Exibir a contagem de linhas do DataFrame
print("Número de linhas em alsides:", len(alsides), "\n")
print("Número de linhas em common :", len(common), "\n")
print("Número de linhas em common 1 (sem duplicados) :", len(common), "\n")
print("Número de linhas em manual_data:", len(manual_data), "\n")
print("Número de linhas em novel:", len(novel), "\n")
print("Número de linhas em trofi:", len(trofi), "\n")
print("Número de linhas em moh:", len(moh), "\n")
print("Número de linhas em metaphor_cot:", len(metaphor_cot), "\n\n")
# print("Início do dataset manual_data:\n")
# print(manual_data.head())


df = pd.read_parquet("comparacao_datasets/common.parquet")

df_cleaned = df.drop_duplicates(subset=["text"])

df_cleaned.to_parquet("comparacao_datasets/common1.parquet", index=False)

