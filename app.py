import streamlit as st
from train_model import train_model
from formulaire_prediction import formulaire_patient
from shap_explainer import get_shap_values
from insert_db import insert_patient
import shap
import matplotlib.pyplot as plt

# Configuration
st.set_page_config(page_title="Application Décès - MLP + SHAP", layout="centered")

# Navigation dans la sidebar
page = st.sidebar.radio("📌 Navigation", ["🏠 Accueil", "📊 Métriques du modèle", "🧾 Prédiction"])

# Accueil
if page == "🏠 Accueil":
    st.title("🧠 Prédiction du Décès chez les Patients - Accueil")
    st.markdown("""
    Bienvenue sur l'application de prédiction du décès chez les patients à partir de caractéristiques cliniques.

    👉 Naviguez via la barre latérale :
    - **📊 Métriques** : Pour consulter les performances du modèle validé  
    - **🧾 Prédiction** : Pour faire une prédiction et interpréter les résultats avec SHAP
    """)

# Page Métriques
elif page == "📊 Métriques du modèle":
    st.title("📊 Métriques validées du modèle")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Accuracy", "97.47 %")
        st.metric("🔍 Précision", "97.78 %")
    with col2:
        st.metric("💡 Sensibilité", "97.78 %")
        st.metric("🛡️ Spécificité", "97.10 %")

    st.caption(" 🎯 Exactitude (Accuracy) - 97.47% L'exactitude mesure la proportion de prédictions correctes. Avec 97.47%, cela signifie que le modèle a correctement prédit si un patient allait décéder ou non dans la majorité des cas. Cela montre une bonne performance générale du modèle.")
    st.caption(" 🔍 Précision (Precision) - 97.78% La précision indique combien de prédictions positives étaient correctes. Avec 97.78%, le modèle prédit correctement la majorité des décès, minimisant les faux positifs. C'est crucial pour éviter des erreurs de classification des décès.")
    st.caption(" 💡 Sensibilité (Recall) - 97.78% La sensibilité mesure la capacité du modèle à détecter les vrais positifs. Ici, 97.78% des décès réels ont été correctement identifiés, ce qui montre que le modèle est très bon pour ne pas manquer de décès importants.")
    st.caption(" 🛡️ Spécificité (Specificity) - 97.10% La spécificité montre la capacité du modèle à identifier les survivants correctement. Avec 97.10%, cela signifie que le modèle a très peu de faux positifs et peut distinguer efficacement les patients survivants des décédés.")

# Page Prédiction + SHAP
elif page == "🧾 Prédiction":
    st.title("🧾 Saisie du patient & interprétation SHAP")

    model, _, _, _, _, X_train = train_model()

    st.markdown("Veuillez remplir les informations cliniques du patient. Toutes les valeurs sont binaires : **1** = Oui, **0** = Non.")
    new_patient = formulaire_patient(X_train.columns)

    if st.button("Prédire et expliquer"):
        prediction = model.predict(new_patient)[0]
        st.markdown(f"## ✅ Prédiction : **{'Décès' if prediction == 1 else 'Survie'}**")

        try:
            ligne = new_patient.iloc[0].tolist()
            ligne.append(int(prediction))
            insert_patient(ligne)
            st.success("📥 Patient enregistré avec succès.")
        except Exception as e:
            st.error(f"❌ Erreur base de données : {e}")

        st.subheader("📊 Interprétation SHAP ")
        try:
            shap_values, explainer = get_shap_values(model, X_train, new_patient)
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values, max_display=11, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Erreur SHAP : {e}")
