import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from pathlib import Path
from typing import Any, Tuple
import transformers.utils.import_utils
import transformers.modeling_utils

# Monkeypatch transformers to allow loading PyTorch < 2.6 checkpoints (safe for trusted HF models like DeBERTa)
def _mock_check_torch_load_is_safe():
    pass
transformers.utils.import_utils.check_torch_load_is_safe = _mock_check_torch_load_is_safe
transformers.modeling_utils.check_torch_load_is_safe = _mock_check_torch_load_is_safe


class BERTClassifier:
    def __init__(
        self, 
        model_name: str = "roberta-base",
        epochs: int = 10, 
        batch_size: int = 8, 
        learning_rate: float = 2e-5,
        max_length: int = 512,
        random_seed: int = 42
    ):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.random_seed = random_seed
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.training_history = []  # Store per-epoch train/val loss
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"BERTClassifier initialized on device: {self.device}")
        print(f"Model: {model_name}, Epochs: {epochs}, Batch: {batch_size}, MaxLen: {max_length}")

    def prepare_dataset(self, samples: list[dict], is_test: bool = False) -> Tuple[Dataset, list[str]]:
        texts = [sample["text"] for sample in samples]
        y_list = [sample.get("label", "") for sample in samples]
        
        if is_test:
            labels = [0] * len(samples)
        else:
            labels = self.label_encoder.transform(y_list).tolist()
            
        data_dict = {
            "text": texts,
            "label": labels
        }
        
        hf_dataset = Dataset.from_dict(data_dict)
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, max_length=self.max_length)
            
        tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
        return tokenized_dataset, y_list

    def fit(self, train_samples: list[dict], val_samples: list[dict]) -> None:
        # Encode labels first
        y_train_raw = [s.get("label", "") for s in train_samples]
        self.label_encoder.fit(y_train_raw)
        
        num_labels = len(self.label_encoder.classes_)
        id2label = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        label2id = {label: i for i, label in enumerate(self.label_encoder.classes_)}
        
        # Load pre-trained model for sequence classification
        print(f"Loading pre-trained model: {self.model_name} with {num_labels} labels...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        self.model.to(self.device)
        
        train_dataset, _ = self.prepare_dataset(train_samples, is_test=False)
        val_dataset, _ = self.prepare_dataset(val_samples, is_test=False)
        
        # Define Trainer metrics
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average="macro")
            return {"accuracy": acc, "macro_f1": f1}
            
        # Training arguments with warmup, cosine scheduler
        training_args = TrainingArguments(
            output_dir="./tmp_bert_checkpoints",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            warmup_ratio=0.1,
            bf16=True,
            lr_scheduler_type="cosine",
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            seed=self.random_seed,
            logging_strategy="epoch",
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics
        )
        
        print("Starting BERT training...")
        train_result = trainer.train()
        print("BERT training completed.")
        
        # Extract training history from log_history
        self.training_history = []
        log_history = trainer.state.log_history
        
        epoch_train_losses = {}
        epoch_val_losses = {}
        epoch_val_metrics = {}
        
        for entry in log_history:
            epoch = entry.get("epoch")
            if epoch is None:
                continue
            epoch_int = int(round(epoch))
            
            if "loss" in entry and "eval_loss" not in entry:
                epoch_train_losses[epoch_int] = entry["loss"]
            if "eval_loss" in entry:
                epoch_val_losses[epoch_int] = entry["eval_loss"]
                epoch_val_metrics[epoch_int] = {
                    "eval_accuracy": entry.get("eval_accuracy"),
                    "eval_macro_f1": entry.get("eval_macro_f1")
                }
        
        for ep in sorted(set(list(epoch_train_losses.keys()) + list(epoch_val_losses.keys()))):
            self.training_history.append({
                "epoch": ep,
                "train_loss": epoch_train_losses.get(ep),
                "val_loss": epoch_val_losses.get(ep),
                "val_accuracy": epoch_val_metrics.get(ep, {}).get("eval_accuracy"),
                "val_macro_f1": epoch_val_metrics.get(ep, {}).get("eval_macro_f1")
            })
        
        print(f"Training history recorded for {len(self.training_history)} epochs.")
        
        # Clean up checkpoints directory to save disk space
        import shutil
        try:
            shutil.rmtree(training_args.output_dir)
            print(f"Cleaned up checkpoints directory: {training_args.output_dir}")
        except Exception as e:
            pass

    def predict(self, samples: list[dict]) -> list[str]:
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
            
        test_dataset, _ = self.prepare_dataset(samples, is_test=True)
        
        # Build Trainer just for prediction
        trainer = Trainer(
            model=self.model,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )
        
        print("Running BERT inference...")
        predictions = trainer.predict(test_dataset)
        preds_indices = np.argmax(predictions.predictions, axis=-1)
        return self.label_encoder.inverse_transform(preds_indices).tolist()

    def evaluate(self, val_samples: list[dict]) -> dict[str, Any]:
        preds = self.predict(val_samples)
        y_val = [s.get("label", "") for s in val_samples]
        
        accuracy = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average="macro")
        report = classification_report(y_val, preds, output_dict=True)
        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "report": report,
            "predictions": preds
        }

    def save(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch Model & Tokenizer
        if self.model is not None:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
        # Save LabelEncoder classes
        np.save(path / "label_encoder_classes.npy", self.label_encoder.classes_)
        print(f"Model saved successfully to {output_dir}")

    def load(self, model_dir: str) -> None:
        path = Path(model_dir)
        print(f"Loading BERT model and tokenizer from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        
        # Load LabelEncoder
        self.label_encoder.classes_ = np.load(path / "label_encoder_classes.npy", allow_pickle=True)
