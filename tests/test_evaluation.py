import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from penguin_lib.models.trainer import PenguinTrainer
from penguin_lib.metrics.evaluation import PenguinEvaluator

def test_evaluation_outputs_metrics():
    trainer = PenguinTrainer()
    X, y = trainer.load_and_prepare("data/penguins.csv")
    results = trainer.train(X, y)

    evaluator = PenguinEvaluator()
    metrics = evaluator.evaluate(results["y_test"], results["y_pred"])

    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "confusion_matrix" in metrics

