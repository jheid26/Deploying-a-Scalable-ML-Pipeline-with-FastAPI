import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model


@pytest.fixture
def sample_data():
    """Load a small deterministic sample of the census data."""
    data = pd.read_csv("data/census.csv")
    return data.sample(n=500, random_state=42)


@pytest.fixture
def processed_split(sample_data):
    """Create train/test matrices and a fitted encoder/label binarizer."""
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    train, test = train_test_split(sample_data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test


def test_train_model_returns_random_forest(processed_split):
    """
    train_model should return a fitted RandomForestClassifier.
    """
    X_train, y_train, _, _ = processed_split
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")
    assert model.n_features_in_ == X_train.shape[1]


def test_inference_output_shape_and_values(processed_split):
    """
    inference should return a 1-D array of binary predictions matching X rows.
    """
    X_train, y_train, X_test, _ = processed_split
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_known_values():
    """
    compute_model_metrics should return exact precision, recall, and F1 on
    a hand-checked example.
    """
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 / 3)
