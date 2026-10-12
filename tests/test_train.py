from src.train import experiment
def test_experiment():
    model,metrics=experiment(150)
    assert 0<=metrics["roc_auc"]<=1
    assert metrics["train_rows"]+metrics["test_rows"]==150
