# Heart Risk ML Lab — Synthetic Data Only

Runnable **educational classification experiment** using scikit-learn. All input features and labels are synthetically generated mathematical samples, **not clinical or patient records**. The model is not a medical device or a real heart-disease predictor.

## Run
```bash
pip install -r requirements.txt
python -m src.train
```
Creates `artifacts/model.joblib` and `artifacts/metrics.json`. Runs train/test splitting, cross-validation on training data, logistic regression versus random forest and holdout ROC-AUC/F1/precision/recall.

## Test
```bash
pip install pytest
pytest -q
```

**Tech:** Python, scikit-learn, machine learning, model selection and evaluation. No clinical validity is claimed.