import shap

def get_shap_values(model, X_train, new_data):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(new_data)
    return shap_values, explainer
