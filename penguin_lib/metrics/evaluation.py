from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

class PenguinEvaluator:
    def evaluate(self, y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="weighted"),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
