import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from penguin_lib.features.feature_creator import FeatureCreator

def test_feature_creator_adds_columns():
    df = pd.DataFrame({
        "bill_length_mm": [40.0],
        "bill_depth_mm": [18.0],
        "flipper_length_mm": [200.0],
        "body_mass_g": [4000],
        "island": ["Dream"],
        "sex": ["Male"],
        "year": [2009]
    })

    creator = FeatureCreator()
    df_new = creator.create_all(df)

    expected = [
        "bill_ratio",
        "mass_flipper_ratio",
        "year_centered",
        "bill_sum",
        "interaction"
    ]

    for col in expected:
        assert col in df_new.columns

