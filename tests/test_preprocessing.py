import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from penguin_lib.preprocessing.preprocessor import PenguinPreprocessor

def test_preprocessor_output_shape():
    df = pd.DataFrame({
        "bill_length_mm": [40.0, 42.0],
        "bill_depth_mm": [18.0, 17.0],
        "flipper_length_mm": [195.0, 210.0],
        "body_mass_g": [3700, 4000],
        "island": ["Biscoe", "Dream"],
        "sex": ["Male", "Female"],
        "year": [2008, 2009]
    })

    pre = PenguinPreprocessor()
    transformed = pre.fit_transform(df)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 0
