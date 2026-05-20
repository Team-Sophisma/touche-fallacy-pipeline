import argparse
import json
from pathlib import Path
from src.infrastructure.models.xgboost_model import XGBoostClassifier
from src.infrastructure.models.bert_model import BERTClassifier
from src.application.services.model_evaluation_service import ModelEvaluationService

def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_feature_map(path: Path) -> dict[str, dict]:
    feature_map = {}
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                item = json.loads(line)
                feature_map[item["id"]] = item
    return feature_map

def write_submission(predictions: list[str], test_samples: list[dict], task: str, model_name: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as writer:
        for pred, sample in zip(predictions, test_samples):
            obj = {
                "task": task,
                "id": sample["id"],
                "label": pred,
                "tag": "enhanced",
                "system_description": f"{model_name}_touche_pipeline"
            }
            writer.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Submission run file saved to: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="CLEF Touché 2026 Model Training and Evaluation Orchestrator")
    parser.add_argument(
        "--task", 
        type=str, 
        default="fallacy_detection", 
        choices=["fallacy_detection", "fallacy_classification"],
        help="Task to run: fallacy_detection (binary) or fallacy_classification (multi-class)"
    )
    parser.add_argument(
        "--models", 
        type=str, 
        nargs="+", 
        default=["xgboost", "bert"],
        help="Models to train: xgboost and/or bert"
    )
    parser.add_argument(
        "--bert_epochs", 
        type=int, 
        default=3,
        help="Number of epochs for BERT fine-tuning"
    )
    
    args = parser.parse_args()
    
    # Paths
    preprocessed_dir = Path("data/processed/preprocessed/enhanced") / args.task
    features_dir = Path("data/processed/features")
    
    print(f"\n=======================================================")
    print(f"Training for Task: {args.task}")
    print(f"=======================================================")
    
    # 1. Load Preprocessed Splits
    print("Loading preprocessed dataset splits...")
    train_samples = load_jsonl(preprocessed_dir / "train.jsonl")
    val_samples = load_jsonl(preprocessed_dir / "validation.jsonl")
    test_samples = load_jsonl(preprocessed_dir / "test.jsonl")
    
    print(f"Splits loaded: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    # 2. Load Extracted Features Map
    print("Loading extracted feature files...")
    train_feature_map = load_feature_map(features_dir / "train_features.jsonl")
    test_feature_map = load_feature_map(features_dir / "test_features.jsonl")
    
    # Combine feature maps since validation IDs are in the train_features.jsonl
    full_feature_map = {**train_feature_map, **test_feature_map}
    
    # 3. Initialize Evaluator
    evaluator = ModelEvaluationService(output_dir="data/reports/model_evaluation")
    
    comparisons = {}
    
    # 4. Train and Evaluate XGBoost
    if "xgboost" in args.models:
        print("\n--- Training XGBoost Model ---")
        xgb_clf = XGBoostClassifier()
        
        print("Preparing feature matrices...")
        X_train, y_train = xgb_clf.prepare_data(train_samples, full_feature_map)
        X_val, y_val = xgb_clf.prepare_data(val_samples, full_feature_map)
        X_test, _ = xgb_clf.prepare_data(test_samples, full_feature_map)
        
        print(f"Training XGBoost on shape: {X_train.shape}...")
        xgb_clf.fit(X_train, y_train, X_val, y_val)
        
        print("Evaluating on validation set...")
        xgb_metrics = xgb_clf.evaluate(X_val, y_val)
        print(f"XGBoost Validation Accuracy: {xgb_metrics['accuracy']:.4f}")
        print(f"XGBoost Validation Macro F1: {xgb_metrics['macro_f1']:.4f}")
        
        # Save report and confusion matrix
        evaluator.generate_report(
            task_name=args.task,
            model_name="xgboost",
            metrics=xgb_metrics,
            y_true=y_val.tolist(),
            y_pred=xgb_metrics["predictions"]
        )
        
        comparisons["XGBoost"] = {
            "accuracy": xgb_metrics["accuracy"],
            "macro_f1": xgb_metrics["macro_f1"]
        }
        
        # Generate Test Predictions and Save Submission
        print("Generating test predictions...")
        xgb_test_preds = xgb_clf.predict(X_test)
        write_submission(
            predictions=xgb_test_preds,
            test_samples=test_samples,
            task=args.task,
            model_name="xgboost",
            output_path=Path("data/submissions") / f"{args.task}_xgboost_submission.jsonl"
        )
        
        # Save model
        xgb_clf.save(f"data/models/{args.task}_xgboost.pkl")
        
    # 5. Train and Evaluate BERT
    if "bert" in args.models:
        print("\n--- Training BERT Model ---")
        bert_clf = BERTClassifier(epochs=args.bert_epochs)
        
        print("Fine-tuning DistilBERT on text...")
        bert_clf.fit(train_samples, val_samples)
        
        print("Evaluating on validation set...")
        bert_metrics = bert_clf.evaluate(val_samples)
        print(f"BERT Validation Accuracy: {bert_metrics['accuracy']:.4f}")
        print(f"BERT Validation Macro F1: {bert_metrics['macro_f1']:.4f}")
        
        y_val = [s.get("label", "") for s in val_samples]
        evaluator.generate_report(
            task_name=args.task,
            model_name="bert",
            metrics=bert_metrics,
            y_true=y_val,
            y_pred=bert_metrics["predictions"]
        )
        
        comparisons["BERT"] = {
            "accuracy": bert_metrics["accuracy"],
            "macro_f1": bert_metrics["macro_f1"]
        }
        
        # Generate Test Predictions and Save Submission
        print("Generating test predictions...")
        bert_test_preds = bert_clf.predict(test_samples)
        write_submission(
            predictions=bert_test_preds,
            test_samples=test_samples,
            task=args.task,
            model_name="bert",
            output_path=Path("data/submissions") / f"{args.task}_bert_submission.jsonl"
        )
        
        # Save model directory
        bert_clf.save(f"data/models/{args.task}_bert")

    # 6. Save Comparison Report if multiple models trained
    if len(comparisons) > 1:
        print("\n--- Saving Comparison Chart ---")
        evaluator.save_comparison(args.task, comparisons)
        
    print("\nTraining and evaluation workflow completed successfully!")

if __name__ == "__main__":
    main()
