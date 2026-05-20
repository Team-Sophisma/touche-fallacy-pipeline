import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
from typing import Any, Tuple

class XGBoostClassifier:
    def __init__(self, n_estimators: int = 150, max_depth: int = 6, learning_rate: float = 0.05, random_seed: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.model = None
        self.label_encoder = LabelEncoder()

    def prepare_data(self, samples: list[dict], feature_map: dict[str, dict]) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []

        for sample in samples:
            sample_id = sample["id"]
            if sample_id not in feature_map:
                continue

            feats = feature_map[sample_id]
            nlp_feats = feats["nlp_features"]
            
            # 1. Stylistic/NLP Features
            row = [
                nlp_feats.get("char_count", 0),
                nlp_feats.get("word_count", 0),
                nlp_feats.get("avg_word_length", 0.0),
                nlp_feats.get("num_sentences", 0),
                nlp_feats.get("q_mark_count", 0),
                nlp_feats.get("excl_mark_count", 0),
                nlp_feats.get("rate_noun", 0.0),
                nlp_feats.get("rate_verb", 0.0),
                nlp_feats.get("rate_adj", 0.0),
                nlp_feats.get("rate_adv", 0.0),
                nlp_feats.get("rate_pron", 0.0),
                nlp_feats.get("rate_propn", 0.0),
                nlp_feats.get("rate_punct", 0.0),
                nlp_feats.get("sentiment_positive", 0.0),
                nlp_feats.get("sentiment_negative", 0.0),
                feats.get("hdbscan_cluster", -1),
                feats.get("hdbscan_outlier_score", 0.0)
            ]
            
            # 2. Add UMAP dimensions
            row.extend(feats.get("umap_2d", [0.0, 0.0]))
            row.extend(feats.get("umap_5d", [0.0, 0.0, 0.0, 0.0, 0.0]))
            
            # 3. Add full 384 embeddings
            row.extend(feats.get("embeddings", [0.0] * 384))
            
            X_list.append(row)
            # Labels may be missing for test set
            y_list.append(sample.get("label", ""))

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        return X, y

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            eval_set = [(X_val, y_val_encoded)]

        num_classes = len(self.label_encoder.classes_)
        objective = "binary:logistic" if num_classes <= 2 else "multi:softprob"
        
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_seed,
            objective=objective,
            eval_metric="mlogloss" if num_classes > 2 else "logloss"
        )
        
        self.model.fit(
            X_train, 
            y_train_encoded,
            eval_set=eval_set,
            verbose=False
        )

    def predict(self, X: np.ndarray) -> list[str]:
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        preds_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds_encoded).tolist()

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
        preds = self.predict(X_val)
        accuracy = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average="macro")
        report = classification_report(y_val, preds, output_dict=True)
        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "report": report,
            "predictions": preds
        }

    def save(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "label_encoder": self.label_encoder
            }, f)

    def load(self, model_path: str) -> None:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.label_encoder = data["label_encoder"]
