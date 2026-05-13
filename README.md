# touche-fallacy-pipeline
A modular and reusable NLP pipeline for the CLEF Touché 2026 Fallacy Detection task, designed for easy experimentation with different datasets, models, and evaluation strategies.

# about the data 
to be able to run the project first both the touchefallacy_2026_train.jsonl, 
touchefallacy_2026_test_task.jsonl files should be downloaded from the https://zenodo.org/records/19925569 website.
Both files should be in the data/raw folders.

# preprocessing

The enhanced-only preprocessing pipeline builds model-ready JSONL files for:

- fallacy_detection
- fallacy_classification

Run it from the project root:

```bash
python run_preprocess.py
```

Outputs are written to:

```text
data/processed/preprocessed/enhanced/
```

The preprocessing configuration is here:

```text
configs/experiment/preprocess_touche_enhanced.yaml
```

Base data and scheme classification can be enabled later by changing the config
instead of rewriting the pipeline.
