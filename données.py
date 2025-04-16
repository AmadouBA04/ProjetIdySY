import sys
import os

# Ajoute le dossier courant (où se trouve ton notebook) dans le PATH
sys.path.append(os.getcwd())

import pandas as pd

# Définition de la fonction
def load_data(file_path="ccc.xlsx", sheet_name="Feuil1"):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    X = df.drop(columns=["DECES"])
    y = df["DECES"]
    return X, y

# Appel de la fonction pour récupérer X et y
X, y = load_data()

# Affichage d'un aperçu
print(X.head())
print(y.head())
