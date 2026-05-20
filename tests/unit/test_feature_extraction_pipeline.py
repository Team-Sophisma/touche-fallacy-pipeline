import pytest
import tempfile
import json
from pathlib import Path
import numpy as np
from src.application.services.feature_extraction_service import FeatureExtractionService

def test_feature_extraction_service():
    # Arrange: Create mock data with at least 15 samples so UMAP/HDBSCAN works
    train_samples = [
        {
            "id": f"train_{i}",
            "text_enhanced": f"This is a fallacious argument sample number {i} containing some words. Fallacies are bad!",
            "sentiment_positive": 0.1 * (i % 5),
            "sentiment_negative": 0.9 - 0.1 * (i % 5)
        } for i in range(15)
    ]
    test_samples = [
        {
            "id": f"test_{i}",
            "text_enhanced": f"This is another test sample number {i} for fallacy detection task.",
            "sentiment_positive": 0.2 * (i % 3),
            "sentiment_negative": 0.8 - 0.2 * (i % 3)
        } for i in range(5)
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        service = FeatureExtractionService(spacy_model="en_core_web_sm", embedding_model="all-MiniLM-L6-v2")
        
        # Act
        service.extract_features(
            train_samples=train_samples,
            test_samples=test_samples,
            output_dir=tmp_dir
        )
        
        # Assert
        train_out = Path(tmp_dir) / "train_features.jsonl"
        test_out = Path(tmp_dir) / "test_features.jsonl"
        
        assert train_out.exists()
        assert test_out.exists()
        
        # Verify schema of train_features
        with open(train_out, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 15
            first_item = json.loads(lines[0])
            
            assert "id" in first_item
            assert "text_enhanced" in first_item
            assert "nlp_features" in first_item
            assert "embeddings" in first_item
            assert "umap_2d" in first_item
            assert "umap_5d" in first_item
            assert "hdbscan_cluster" in first_item
            assert "hdbscan_outlier_score" in first_item
            
            # Verify nlp_features keys
            nlp_feats = first_item["nlp_features"]
            assert "char_count" in nlp_feats
            assert "word_count" in nlp_feats
            assert "avg_word_length" in nlp_feats
            assert "num_sentences" in nlp_feats
            assert "rate_noun" in nlp_feats
            assert "rate_verb" in nlp_feats
            assert "rate_adj" in nlp_feats
            assert "rate_punct" in nlp_feats
            assert "sentiment_positive" in nlp_feats
            assert "sentiment_negative" in nlp_feats
            
            # Check dimensions
            assert len(first_item["embeddings"]) == 384
            assert len(first_item["umap_2d"]) == 2
            assert len(first_item["umap_5d"]) == 5

        # Verify schema of test_features
        with open(test_out, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 5
            first_item = json.loads(lines[0])
            assert "id" in first_item
            assert "nlp_features" in first_item
            assert len(first_item["embeddings"]) == 384
            assert len(first_item["umap_2d"]) == 2
            assert len(first_item["umap_5d"]) == 5
