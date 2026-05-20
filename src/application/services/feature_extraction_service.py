import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import umap
import torch
import json
from pathlib import Path
from tqdm import tqdm
from typing import Sequence, Any, Dict

class FeatureExtractionService:
    def __init__(self, spacy_model: str = "en_core_web_sm", embedding_model: str = "all-MiniLM-L6-v2"):
        print(f"Loading spaCy model: {spacy_model}")
        self.nlp = spacy.load(spacy_model)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SentenceTransformer model: {embedding_model} on device: {device}")
        self.embedder = SentenceTransformer(embedding_model, device=device)

    def extract_features(self, train_samples: list[dict], test_samples: list[dict], output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Extracting NLP and stylistic features...")
        train_features = self._extract_basic_and_spacy_features(train_samples)
        test_features = self._extract_basic_and_spacy_features(test_samples)

        print("Computing contextual embeddings...")
        train_embeddings = self._compute_embeddings(train_samples)
        test_embeddings = self._compute_embeddings(test_samples)

        # Apply UMAP
        print("Reducing dimensions with UMAP...")
        # Fit UMAP on train embeddings, transform both train and test
        # We project to 2D for visualization, and 5D for HDBSCAN clustering
        umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_5d = umap.UMAP(n_components=5, random_state=42, n_neighbors=15, min_dist=0.0)

        print("Fitting UMAP 2D...")
        train_umap_2d = umap_2d.fit_transform(train_embeddings)
        test_umap_2d = umap_2d.transform(test_embeddings)

        print("Fitting UMAP 5D...")
        train_umap_5d = umap_5d.fit_transform(train_embeddings)
        test_umap_5d = umap_5d.transform(test_embeddings)

        # HDBSCAN clustering on UMAP 5D embeddings
        print("Clustering with HDBSCAN...")
        # Combine train and test to discover global clusters and outliers
        all_umap_5d = np.vstack([train_umap_5d, test_umap_5d])
        hdb = HDBSCAN(min_cluster_size=10, min_samples=5, store_centers='centroid')
        cluster_labels = hdb.fit_predict(all_umap_5d)
        outlier_scores = 1.0 - hdb.probabilities_

        train_len = len(train_samples)
        train_clusters = cluster_labels[:train_len]
        test_clusters = cluster_labels[train_len:]
        
        train_outliers = outlier_scores[:train_len]
        test_outliers = outlier_scores[train_len:]

        # Merge all features
        print("Merging all features...")
        final_train_data = self._merge_features(
            train_samples, train_features, train_embeddings, 
            train_umap_2d, train_umap_5d, train_clusters, train_outliers
        )
        final_test_data = self._merge_features(
            test_samples, test_features, test_embeddings, 
            test_umap_2d, test_umap_5d, test_clusters, test_outliers
        )

        # Write outputs
        train_out_file = output_path / "train_features.jsonl"
        test_out_file = output_path / "test_features.jsonl"

        self._write_jsonl(final_train_data, train_out_file)
        self._write_jsonl(final_test_data, test_out_file)

        print(f"Features saved successfully.\nTrain: {train_out_file}\nTest: {test_out_file}")

    def _extract_basic_and_spacy_features(self, samples: list[dict]) -> list[dict]:
        features_list = []
        for sample in tqdm(samples, desc="NLP Processing"):
            text = sample.get("text_enhanced", "")
            
            # Basic stats
            char_count = len(text)
            words = text.split()
            word_count = len(words)
            avg_word_length = np.mean([len(w) for w in words]) if word_count > 0 else 0.0

            # spaCy analysis
            doc = self.nlp(text)
            num_sentences = len(list(doc.sents))

            # POS tag rates
            pos_counts = {
                "NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0, "PRON": 0, "PROPN": 0, "PUNCT": 0
            }
            for token in doc:
                if token.pos_ in pos_counts:
                    pos_counts[token.pos_] += 1

            # Normalize POS counts by word count
            pos_rates = {}
            for pos, count in pos_counts.items():
                pos_rates[f"rate_{pos.lower()}"] = count / word_count if word_count > 0 else 0.0

            # Special punctuation frequencies
            q_mark_count = text.count("?")
            excl_mark_count = text.count("!")
            
            # Combine all features
            feats = {
                "char_count": char_count,
                "word_count": word_count,
                "avg_word_length": avg_word_length,
                "num_sentences": num_sentences,
                "q_mark_count": q_mark_count,
                "excl_mark_count": excl_mark_count,
                **pos_rates,
                "sentiment_positive": sample.get("sentiment_positive", 0.0),
                "sentiment_negative": sample.get("sentiment_negative", 0.0)
            }
            features_list.append(feats)
        return features_list

    def _compute_embeddings(self, samples: list[dict]) -> np.ndarray:
        texts = [sample.get("text_enhanced", "") for sample in samples]
        embeddings = self.embedder.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        return embeddings

    def _merge_features(
        self, 
        samples: list[dict], 
        features: list[dict], 
        embeddings: np.ndarray,
        umap_2d: np.ndarray, 
        umap_5d: np.ndarray, 
        clusters: np.ndarray, 
        outliers: np.ndarray
    ) -> list[dict]:
        merged = []
        for i, sample in enumerate(samples):
            item = dict(sample)
            
            # Add structural / NLP features
            item["nlp_features"] = features[i]
            
            # Add embeddings
            item["embeddings"] = embeddings[i].tolist()
            
            # Add UMAP coordinates
            item["umap_2d"] = umap_2d[i].tolist()
            item["umap_5d"] = umap_5d[i].tolist()
            
            # Add HDBSCAN cluster and outlier score
            item["hdbscan_cluster"] = int(clusters[i])
            item["hdbscan_outlier_score"] = float(outliers[i])
            
            merged.append(item)
        return merged

    def _write_jsonl(self, data: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as writer:
            for item in data:
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")
