import os
import json
import mlflow
import numpy as np
import pandas as pd
import pytest
from src.train import build_model, train


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    """
    Tao dataset nho voi cung schema Wine Quality de su dung trong test.

    pytest cung cap `tmp_path` la mot thu muc tam thoi, tu dong xoa sau khi test ket thuc.
    Ham nay dung du lieu ngau nhien nen khong can ket noi GCS hay tai file CSV thuc.
    """
    rng = np.random.default_rng(0)
    n = 200

    X = rng.random((n, len(FEATURE_NAMES)))

    y = rng.integers(0, 3, size=n)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    return train_path, eval_path


def _make_imbalanced_temp_data(tmp_path):
    """Tao dataset co lop 2 duoi 10% de kiem tra canh bao drift."""
    rng = np.random.default_rng(1)
    n = 120

    X = rng.random((n, len(FEATURE_NAMES)))
    y = np.array([0] * 60 + [1] * 55 + [2] * 5)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_path = str(tmp_path / "imbalanced_train.csv")
    eval_path = str(tmp_path / "imbalanced_eval.csv")
    df.iloc[:100].to_csv(train_path, index=False)
    df.iloc[100:].to_csv(eval_path, index=False)

    return train_path, eval_path


def _use_temp_mlflow(tmp_path):
    """Tach MLflow test runs khoi thu muc mlruns cua repo."""
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())


def _params(model_type="random_forest"):
    """Tao params dung schema moi cua Bonus 2 cho cac test nho."""
    return {
        "model_type": model_type,
        "random_forest": {
            "n_estimators": 10,
            "max_depth": 3,
            "min_samples_split": 2,
        },
        "gradient_boosting": {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "max_depth": 2,
        },
        "logistic_regression": {
            "C": 1.0,
            "max_iter": 200,
        },
    }


def test_train_returns_float(tmp_path):
    """Kiem tra ham train() tra ve mot so thuc nam trong [0.0, 1.0]."""
    _use_temp_mlflow(tmp_path)
    train_path, eval_path = _make_temp_data(tmp_path)

    acc = train(
        _params("random_forest"),
        data_path=train_path,
        eval_path=eval_path,
    )

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    """Kiem tra file outputs/metrics.json duoc tao sau khi huan luyen."""
    _use_temp_mlflow(tmp_path)
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        _params("random_forest"),
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "label_distribution" in metrics
    assert set(metrics["label_distribution"]) == {"0", "1", "2"}


def test_model_file_created(tmp_path):
    """Kiem tra file models/model.pkl duoc tao sau khi huan luyen."""
    _use_temp_mlflow(tmp_path)
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        _params("random_forest"),
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("models/model.pkl")


def test_report_file_created(tmp_path):
    """Kiem tra report.txt co confusion matrix, precision va recall."""
    _use_temp_mlflow(tmp_path)
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        _params("random_forest"),
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/report.txt")
    with open("outputs/report.txt") as f:
        report = f.read()
    assert "Confusion Matrix" in report
    assert "precision" in report
    assert "recall" in report


def test_imbalanced_labels_print_warning(tmp_path, capsys):
    """Kiem tra canh bao khi mot lop chiem duoi 10% tap train."""
    _use_temp_mlflow(tmp_path)
    train_path, eval_path = _make_imbalanced_temp_data(tmp_path)
    train(
        _params("random_forest"),
        data_path=train_path,
        eval_path=eval_path,
    )

    captured = capsys.readouterr()
    assert "Label 2 ratio:" in captured.out
    assert "WARNING: label 2 ratio" in captured.out
    assert "Possible data drift." in captured.out


def test_train_supports_gradient_boosting(tmp_path):
    """Kiem tra co the huan luyen voi GradientBoostingClassifier."""
    _use_temp_mlflow(tmp_path)
    train_path, eval_path = _make_temp_data(tmp_path)

    acc = train(
        _params("gradient_boosting"),
        data_path=train_path,
        eval_path=eval_path,
    )

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_unsupported_model_type_raises_error():
    """Kiem tra model_type khong ho tro tra ve loi ro rang."""
    params = _params("svm")

    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_model(params)
