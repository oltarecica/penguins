from penguin_lib.models.trainer import PenguinTrainer
from penguin_lib.metrics.evaluation import PenguinEvaluator

trainer = PenguinTrainer()
X, y = trainer.load_and_prepare("data/penguins.csv")
results = trainer.train(X, y)

evaluator = PenguinEvaluator()
metrics = evaluator.evaluate(results["y_test"], results["y_pred"])

print(metrics)
