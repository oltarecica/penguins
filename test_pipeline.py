import pandas as pd
from penguin_lib.preprocessing.preprocessor import PenguinPreprocessor
from penguin_lib.features.feature_creator import FeatureCreator

# Load dataset
df = pd.read_csv("data/penguins.csv")

# Create features
fc = FeatureCreator()
df_features = fc.create_all(df)

print("Feature columns added:")
print([col for col in df_features.columns if col not in df.columns])

# Preprocess
prep = PenguinPreprocessor()
X = prep.fit_transform(df_features)

print("Preprocessing output shape:", X.shape)
