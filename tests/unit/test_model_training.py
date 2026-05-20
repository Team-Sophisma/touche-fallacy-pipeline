import pytest
import tempfile
import numpy as np
from pathlib import Path
from src.infrastructure.models.xgboost_model import XGBoostClassifier

def test_xgboost_classifier_workflow():
    # Arrange
    train_samples = [
        {"id": f"s_{i}", "label": "fallacy" if i % 2 == 0 else "non-fallacy"} for i in range(10)
    ]
    val_samples = [
        {"id": f"v_{i}", "label": "fallacy" if i % 2 == 0 else "non-fallacy"} for i in range(4)
    ]
    test_samples = [
        {"id": f"t_{i}"} for i in range(3)
    ]

    # Create dummy features map
    feature_map = {}
    # Train
    for sample in train_samples + val_samples + test_samples:
        feature_map[sample["id"]] = {
            "nlp_features": {
                "char_count": 100,
                "word_count": 20,
                "avg_word_length": 5.0,
                "num_sentences": 2,
                "q_mark_count": 0,
                "excl_mark_count": 0,
                "rate_noun": 0.2,
                "rate_verb": 0.1,
                "rate_adj": 0.1,
                "rate_adv": 0.05,
                "rate_pron": 0.05,
                "rate_propn": 0.05,
                "rate_punct": 0.1,
                "sentiment_positive": 0.5 if "s" in sample["id"] and int(sample["id"].split("_")[1]) % 2 == 0 else 0.1,
                "sentiment_negative": 0.1 if "s" in sample["id"] and int(sample["id"].split("_")[1]) % 2 == 0 else 0.5
            },
            "umap_2d": [1.0, 2.0],
            "umap_5d": [1.0, 2.0, 3.0, 4.0, 5.0],
            "hdbscan_cluster": 1 if "s" in sample["id"] else 0,
            "hdbscan_outlier_score": 0.1,
            "embeddings": [0.05] * 384
        }

    clf = XGBoostClassifier(n_estimators=10, max_depth=3)
    
    # Act: Prepare data
    X_train, y_train = clf.prepare_data(train_samples, feature_map)
    X_val, y_val = clf.prepare_data(val_samples, feature_map)
    X_test, _ = clf.prepare_data(test_samples, feature_map)

    # Assert shapes
    assert X_train.shape == (10, 17 + 2 + 5 + 384)  # 17 tabular feats + 2 umap + 5 umap + 384 embeddings = 408
    assert y_train.shape == (10,)
    assert X_test.shape == (3, 408)

    # Act: Train
    clf.fit(X_train, y_train, X_val, y_val)
    
    # Act: Predict & Evaluate
    preds = clf.predict(X_val)
    assert len(preds) == 4
    assert all(p in ["fallacy", "non-fallacy"] for p in preds)

    metrics = clf.evaluate(X_val, y_val)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "report" in metrics

    # Act: Save and Load
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "xgboost_model.pkl"
        clf.save(str(model_path))
        assert model_path.exists()

        new_clf = XGBoostClassifier()
        new_clf.load(str(model_path))
        
        # Test loaded model prediction
        new_preds = new_clf.predict(X_val)
        assert new_preds == preds
