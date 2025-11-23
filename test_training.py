from penguin_lib.models.trainer import PenguinTrainer

trainer = PenguinTrainer()

X, y = trainer.load_and_prepare("data/penguins.csv")
results = trainer.train(X, y)

print(results)
