import yaml
import json
from pathlib import Path
from src.application.services.feature_extraction_service import FeatureExtractionService

def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                data.append(json.loads(line))
    return data

def main() -> None:
    config_path = "configs/preprocessing/feature_extraction.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    train_path = config["inputs"]["train_enriched"]
    test_path = config["inputs"]["test_enriched"]
    output_dir = config["outputs"]["output_dir"]

    print(f"Reading enriched datasets:\nTrain: {train_path}\nTest: {test_path}")
    train_samples = load_jsonl(train_path)
    test_samples = load_jsonl(test_path)

    print(f"Loaded {len(train_samples)} train samples and {len(test_samples)} test samples.")

    service = FeatureExtractionService(
        spacy_model=config.get("spacy_model", "en_core_web_sm"),
        embedding_model=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    )

    service.extract_features(
        train_samples=train_samples,
        test_samples=test_samples,
        output_dir=output_dir
    )

    print("\nFeature extraction completed successfully!")

if __name__ == "__main__":
    main()
