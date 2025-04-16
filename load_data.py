import pandas as pd

def load_data(file_path="ccc.xlsx", sheet_name="Feuil1"):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    X = df.drop(columns=["DECES"])
    y = df["DECES"]
    return X, y
