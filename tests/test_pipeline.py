import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from penguin_lib.preprocessing.preprocessor import PenguinPreprocessor
from penguin_lib.features.feature_creator import FeatureCreator

def test_pipeline_runs_end_to_end():
    df = pd.read_csv("data/penguins.csv")

    fc = FeatureCreator()
    df_features = fc.create_all(df)

    prep = PenguinPreprocessor()
    X = prep.fit_transform(df_features)

    assert X.shape[0] == df.shape[0]
    assert X.shape[1] > 0

