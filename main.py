import pandas as pd
import joblib
import os

if __name__ == "__main__":
    model_path = os.path.join("models", "best_random_forest_model.pkl")
    model = joblib.load(model_path)

    test_df = pd.read_csv("TEST.csv")

    X_test = test_df.drop(columns=["ID"])

    predictions = model.predict(X_test)

    output_df = pd.DataFrame({
        "ID": test_df['ID'],
        "Class": predictions
    })

    output_df.to_csv("output/output.csv", index=False)

    print("Prediction file saved as output.csv")