import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from penguin_lib.preprocessing.preprocessor import PenguinPreprocessor
from penguin_lib.features.feature_creator import FeatureCreator


class PenguinTrainer:
    def __init__(self):
        self.preprocessor = PenguinPreprocessor()
        self.feature_creator = FeatureCreator()
        self.best_model = None

    def load_and_prepare(self, path: str):
        df = pd.read_csv(path)

        # Feature engineering
        df = self.feature_creator.create_all(df)

        # Separate
        X = df.drop(columns=["species"])
        y = df["species"]

        return X, y

    def train(self, X, y):
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Preprocess
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)

        # Base model
        model = RandomForestClassifier(random_state=42)

        # Hyperparameter tuning
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [0.1, 0.05, 0.005],
            "min_samples_split": [2, 5],
        }

        grid = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1
        )

        grid.fit(X_train_transformed, y_train)

        self.best_model = grid.best_estimator_

        # Evaluate on test set
        test_score = self.best_model.score(X_test_transformed, y_test)

        # Predictions for evaluation
        y_pred = self.best_model.predict(X_test_transformed)

        return {
            "best_params": grid.best_params_,
            "test_accuracy": test_score,
            "y_test": y_test,
            "y_pred": y_pred
        }
