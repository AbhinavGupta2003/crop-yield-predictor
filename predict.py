import joblib

def predict_yield(input_df, model_choice="Stacked"):
    model = joblib.load("models/stack_model.pkl")
    return model.predict(input_df)[0]