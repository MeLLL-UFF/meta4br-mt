import pandas as pd
import os

root_dir = "."

paths = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == "dataset_final_conservador.csv":
            full_path = os.path.join(dirpath, filename)
            paths.append(full_path)

for path in paths:
    df = pd.read_csv(path)
    df = df.sort_values(by="score_combinado", ascending=False)
    path_sem_ext = path[:-4] if path.lower().endswith('.csv') else path
    df.to_csv(f"{path_sem_ext}_ordenado.csv", index=False)