import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
from pathlib import Path
from typing import Any

class ModelEvaluationService:
    def __init__(self, output_dir: str = "data/reports/model_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self, 
        task_name: str, 
        model_name: str, 
        metrics: dict[str, Any], 
        y_true: list[str], 
        y_pred: list[str]
    ) -> None:
        # Save metrics as JSON
        json_report_path = self.output_dir / f"{task_name}_{model_name}_metrics.json"
        with open(json_report_path, "w", encoding="utf-8") as f:
            # We filter predictions to avoid huge files, saving only general metrics
            summary_metrics = {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "classification_report": metrics["report"]
            }
            json.dump(summary_metrics, f, indent=2)

        # Plot and save Confusion Matrix
        labels = sorted(list(set(y_true)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=labels, 
            yticklabels=labels,
            cbar=False
        )
        plt.title(f"Confusion Matrix - {task_name} ({model_name})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        
        plot_path = self.output_dir / f"{task_name}_{model_name}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"[{model_name}] Metrics saved to: {json_report_path}")
        print(f"[{model_name}] Confusion matrix plot saved to: {plot_path}")

    def save_comparison(self, task_name: str, comparisons: dict[str, dict[str, float]]) -> None:
        # comparisons: { "XGBoost": {"accuracy": 0.8, "macro_f1": 0.78}, "BERT": {"accuracy": 0.85, ...} }
        comp_path = self.output_dir / f"{task_name}_model_comparison.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparisons, f, indent=2)
            
        print(f"Model comparison saved to: {comp_path}")
        
        # Plot model comparison bar chart
        models = list(comparisons.keys())
        accuracy_scores = [comparisons[m]["accuracy"] for m in models]
        f1_scores = [comparisons[m]["macro_f1"] for m in models]
        
        x = range(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 6))
        rects1 = ax.bar([i - width/2 for i in x], accuracy_scores, width, label='Accuracy', color='#4f81bd')
        rects2 = ax.bar([i + width/2 for i in x], f1_scores, width, label='Macro F1', color='#c0504d')
        
        ax.set_ylabel('Scores')
        ax.set_title(f'Model Comparison - {task_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.05)
        ax.legend()
        
        # Add labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        plot_path = self.output_dir / f"{task_name}_model_comparison.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Model comparison plot saved to: {plot_path}")
