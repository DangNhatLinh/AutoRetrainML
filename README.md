# AutoTrainML

A small, configuration‑driven pipeline that takes you from raw tabular data to a trained model with minimal fuss. No orchestration layers, no heavy frameworks — just a single YAML file and a few plain‑English commands.

## What it does

* Loads your data and performs basic preprocessing (impute, scale, encode).
* Splits into train/validation/test with stratification when needed.
* Trains scikit‑learn–compatible models (e.g., XGBoost/LightGBM/LogReg/RandomForest).
* Optionally runs simple hyperparameter tuning.
* Evaluates with common metrics and saves a short report.
* Exports the final model (pickle; ONNX optional) together with preprocessing.

## Why it’s useful

* Consistent results across runs via a single declarative config (`autotrain.yml`).
* Fast to set up for new datasets or quick baselines.
* Lightweight: runs locally without Airflow, Docker, PyTorch, or MLflow.

## How it works

1. You describe your dataset, target column, splits, model, and metrics in `autotrain.yml`.
2. The pipeline reads that file, prepares the data, trains, evaluates, and writes outputs to an `experiments/` folder.
3. A compact summary (parameters, metrics, and artifacts) is saved so you can compare runs.

## Files & folders

* `autotrain.yml` — project configuration you edit for each dataset.
* `data/` — raw and processed data (you decide the layout).
* `experiments/` — per‑run logs, reports, and exported models.
* `src/autotrainml_ext/` — optional hooks for custom datasets, models, or metrics.

## Configuration highlights

* Data: path, format (CSV/Parquet), target column, features, and split strategy.
* Preprocessing: imputation, scaling, and encoding options.
* Model: pick a scikit‑learn–compatible estimator and set its parameters.
* Metrics: choose a primary metric (e.g., F1, ROC‑AUC, RMSE) plus any extras.
* Tuning (optional): number of trials and a simple search space.
* Export: artifact formats and whether to include preprocessing.

## Outputs

* A short metrics report for validation and/or test.
* A versioned folder with the trained model and preprocessing steps.
* Plain CSV/JSON logs for easy diffing or spreadsheets.

## Limitations

* Focused on tabular problems. Vision or NLP pipelines are out of scope here.
* Not meant to replace full MLOps stacks; it’s a quick, local workflow.

## License

MIT. See `LICENSE` for details.
