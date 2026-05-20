import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
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

class BERTClassifier:
    def __init__(self, model_name: str = "distilbert-base-uncased", epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5, random_seed: int = 42):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"BERTClassifier initialized on device: {self.device}")

    def prepare_dataset(self, samples: list[dict], is_test: bool = False) -> Tuple[Dataset, list[str]]:
        texts = [sample["text"] for sample in samples]
        y_list = [sample.get("label", "") for sample in samples]
        
        if is_test:
            # We don't use label encoding for test, labels are empty/unlabeled
            labels = [0] * len(samples)
        else:
            labels = self.label_encoder.transform(y_list).tolist()
            
        data_dict = {
            "text": texts,
            "label": labels
        }
        
        hf_dataset = Dataset.from_dict(data_dict)
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, max_length=512)
            
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
            
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./tmp_bert_checkpoints",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            seed=self.random_seed,
            logging_steps=10,
            report_to="none" # disable weights and biases etc.
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics
        )
        
        print("Starting BERT training...")
        trainer.train()
        print("BERT training completed.")

    def predict(self, samples: list[dict]) -> list[str]:
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
            
        test_dataset, _ = self.prepare_dataset(samples, is_test=True)
        
        # Build Trainer just for prediction
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
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
