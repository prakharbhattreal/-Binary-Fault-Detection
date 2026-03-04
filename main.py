import pandas as pd
import joblib
import os

if __name__ == "__main__":
    column_path = os.path.join("models", "columns.pkl")
    model_path = os.path.join("models", "best_model.pkl")

    columns=joblib.load(column_path)
    model = joblib.load(model_path)

    test_df = pd.read_csv("TEST.csv")

    X_test = test_df.drop(columns=["ID"])
    X_test = X_test[columns]

    predictions = model.predict(X_test)

    output_df = pd.DataFrame({
        "ID": test_df['ID'],
        "CLASS": predictions
    })

    output_df.to_csv("FINAL.csv", index=False)

    print("Prediction file saved as FINAL.csv")