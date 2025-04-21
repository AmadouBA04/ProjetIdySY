import shap

def get_shap_values(model, X_train, new_data):
    # Sélectionne un échantillon de fond pour l'explication
    background = X_train.sample(100, random_state=42)
    
    # Crée un explainer basé sur les probabilités du modèle
    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    # Calcule les valeurs SHAP pour la nouvelle donnée
    shap_values = explainer.shap_values(new_data)
    
    # Vérification pour gérer les modèles multi-classe ou à plusieurs sorties
    if isinstance(shap_values, list) and len(shap_values) > 1:
        # Si plusieurs classes, utilisez la classe d'intérêt (par exemple, la classe 1)
        shap_val = shap_values[1][0]  # Index de la classe positive
        base_val = explainer.expected_value[1]
    else:
        # Si une seule classe (binaire), utilisez la classe 0
        shap_val = shap_values[0][0]
        base_val = explainer.expected_value[0]

    # Créer l'objet d'explication SHAP
    explanation = shap.Explanation(
        values=shap_val,
        base_values=base_val,
        data=new_data.iloc[0],
        feature_names=new_data.columns
    )

    return explanation, explainer
