import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EVAL_THRESHOLD = 0.70
DRIFT_WARNING_THRESHOLD = 0.10
LABELS = [0, 1, 2]


def build_model(params: dict):
    """
    Tao mo hinh theo model_type trong params.yaml.

    Tra ve:
        model: mo hinh scikit-learn da cau hinh.
        model_type: ten thuat toan dung de log len MLflow.
        model_params: sieu tham so rieng cua thuat toan da chon.
    """
    model_type = params.get("model_type", "random_forest")
    model_params = params.get(model_type, {})

    if model_type == "random_forest":
        model = RandomForestClassifier(**model_params, random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(**model_params, random_state=42)
    elif model_type == "logistic_regression":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(**model_params, random_state=42),
                ),
            ]
        )
    else:
        raise ValueError(
            "Unsupported model_type. Choose one of: "
            "random_forest, gradient_boosting, logistic_regression"
        )

    return model, model_type, model_params


def write_performance_report(
    model_type: str,
    accuracy: float,
    f1: float,
    y_true,
    y_pred,
    report_path: str = "outputs/report.txt",
) -> None:
    """Ghi bao cao hieu suat dang text cho Bonus 3."""
    labels = [0, 1, 2]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    class_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    lines = [
        "Model Performance Report",
        f"model_type: {model_type}",
        f"accuracy: {accuracy:.4f}",
        f"f1_score: {f1:.4f}",
        "",
        "Confusion Matrix",
        "labels: 0, 1, 2",
    ]
    lines.extend(" ".join(str(value) for value in row) for row in matrix)
    lines.extend(
        [
            "",
            "Classification Report",
            class_report,
        ]
    )

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def check_label_distribution(y_train) -> dict:
    """Tinh ty le nhan va in canh bao neu co lop qua it mau."""
    total = len(y_train)
    label_distribution = {}

    for label in LABELS:
        ratio = float((y_train == label).sum() / total) if total else 0.0
        label_distribution[str(label)] = ratio
        print(f"Label {label} ratio: {ratio:.4f}")
        if ratio < DRIFT_WARNING_THRESHOLD:
            print(
                f"WARNING: label {label} ratio {ratio:.4f} is below "
                f"{DRIFT_WARNING_THRESHOLD:.2f}. Possible data drift."
            )

    return label_distribution


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua model_type va sieu tham so cho tung thuat toan.
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

        # Tao mo hinh theo model_type de so sanh nhieu thuat toan trong Bonus 2.
        model, model_type, model_params = build_model(params)

        # Luu lai thuat toan va sieu tham so da dung de so sanh tren MLflow.
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)

        # Kiem tra phan phoi nhan truoc khi huan luyen de phat hien data drift.
        label_distribution = check_label_distribution(y_train)
        for label, ratio in label_distribution.items():
            mlflow.log_metric(f"label_ratio_{label}", ratio)

        # random_state giup ket qua co tinh tai tao trong lab va CI.
        model.fit(X_train, y_train)

        # Danh gia mo hinh tren tap eval doc lap voi tap huan luyen.
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # Tao bao cao text gom confusion matrix, precision va recall cho Bonus 3.
        write_performance_report(model_type, acc, f1, y_eval, preds)

        # Ghi chi so va artifact mo hinh vao MLflow.
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("outputs/report.txt")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Luu metrics ra file de GitHub Actions doc o cac buoc CI/CD tiep theo.
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump(
                {
                    "accuracy": acc,
                    "f1_score": f1,
                    "label_distribution": label_distribution,
                },
                f,
            )

        # Luu model dang pickle de job deploy co the upload len cloud storage.
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
