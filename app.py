import streamlit as st
from train_model import train_model
from formulaire_prediction import formulaire_patient
from shap_explainer import get_shap_values
from insert_db import insert_patient
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prédiction Décès - SHAP", layout="centered")
st.title("🧠Prédiction du Décès chez les Patients - Modèle Interprétable🧠")
st.markdown("""
Bienvenue sur l'application de prédiction du décès chez les patients à partir de caractéristiques cliniques.  
Cette application utilise un modèle d'apprentissage automatique interprétable par SHAP pour :
- Prédire le risque de décès
- Expliquer les facteurs qui influencent la prédiction
""")

# Charger le modèle et les données
model, accuracy, X_train = train_model()

# Afficher l'accuracy globale
st.markdown(f"### 🎯 Précision actuelle du modèle : **{accuracy:.2%}**")
st.caption("Ce score indique la proportion de prédictions correctes réalisées par le modèle sur les données d'entraînement.")

# Formulaire Streamlit
st.header("🧾 Saisie des caractéristiques du patient")
st.markdown("Veuillez remplir les informations cliniques du patient. Toutes les valeurs sont binaires : **1** pour Oui / Présent, **0** pour Non / Absent.")
new_patient = formulaire_patient(X_train.columns)

# Si l'utilisateur clique sur le bouton
if st.button("Prédire et expliquer"):
    prediction = model.predict(new_patient)[0]
    st.markdown(f"## ✅ Prédiction : **{'Décès' if prediction == 1 else 'Survie'}**")

    # Enregistrement dans la base de données
    try:
        ligne = new_patient.iloc[0].tolist()
        ligne.append(int(prediction))  # Ajout de la colonne DECES
        insert_patient(ligne)
        st.success("📥 Le patient a été enregistré dans la base de données avec succès.")
    except Exception as e:
        st.error(f"❌ Erreur base de données : {e}")

    # Valeurs SHAP
    shap_values, explainer = get_shap_values(model, X_train, new_patient)

    st.subheader("📊 Interprétation des variables (SHAP)")

    # Pour modèle binaire, afficher l’explication de la prédiction classe 1 (Décès)
    try:
        shap.plots.waterfall(shap_values[0][:, 1], show=False)
        fig = plt.gcf()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erreur SHAP : {e}")
        st.subheader("📊 Explication de la prédiction (SHAP)")
    st.markdown("""
    Le graphique ci-dessus montre l'influence de chaque variable sur la prédiction.  
    Les barres rouges augmentent le risque de décès, tandis que les barres bleues le réduisent.
    """)


    
