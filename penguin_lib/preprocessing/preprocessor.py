import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class PenguinPreprocessor:
    def __init__(self):
        self.numeric_features = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]

        self.categorical_features = [
            "island",
            "sex",
            "year",
        ]

        self.preprocessor = self._build_preprocessor()

    def _build_preprocessor(self):
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df = df.drop(columns=["id", "species"], errors="ignore")
        self.preprocessor.fit(df)
        return self

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        df = df.drop(columns=["id", "species"], errors="ignore")
        return self.preprocessor.transform(df)

    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()
        df = df.drop(columns=["id", "species"], errors="ignore")
        return self.preprocessor.fit_transform(df)
