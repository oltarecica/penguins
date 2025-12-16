import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from penguin_lib.models.trainer import PenguinTrainer

def test_training_runs_and_returns_results():
    trainer = PenguinTrainer()
    X, y = trainer.load_and_prepare("data/penguins.csv")
    results = trainer.train(X, y)

    assert "best_params" in results
    assert "test_accuracy" in results
    assert results["test_accuracy"] >= 0

