import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    # Doc du lieu huan luyen va danh gia tu cac file CSV dau vao.
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Tach cot target lam nhan, cac cot con lai la dac trung dau vao.
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Moi lan huan luyen duoc boc trong mot MLflow run de theo doi thi nghiem.
    with mlflow.start_run():

        # Luu lai sieu tham so de co the so sanh giua cac lan chay.
        mlflow.log_params(params)

        # random_state giup ket qua co tinh tai tao trong lab va CI.
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Danh gia mo hinh tren tap eval doc lap voi tap huan luyen.
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # Ghi chi so va artifact mo hinh vao MLflow.
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Luu metrics ra file de GitHub Actions doc o cac buoc CI/CD tiep theo.
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        # Luu model dang pickle de job deploy co the upload len cloud storage.
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
