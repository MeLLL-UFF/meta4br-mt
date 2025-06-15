import pandas as pd

df = pd.read_csv("comparacao_datasets/newsmet.csv")
count_label_1 = (df['predicted_label'] == "metaphorical").sum()
print(f"Total de linhas metafóricas no newsmet: {count_label_1} de {len(df)}, cerca de {count_label_1/len(df)*100:.2f}%")

df = pd.read_parquet("comparacao_datasets/manual_data.parquet")
count_label_1 = (df['Label'] == 1).sum()
print(f"Total de linhas metafóricas no manualdata: {count_label_1} de {len(df)}, cerca de {count_label_1/len(df)*100:.2f}%")

df = pd.read_parquet("comparacao_datasets/common1.parquet")
count_label_1 = (df['label'] == 1).sum()
print(f"Total de linhas metafóricas no common1: {count_label_1} de {len(df)}, cerca de {count_label_1/len(df)*100:.2f}%")