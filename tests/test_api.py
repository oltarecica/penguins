import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from penguin_lib.api.predict import predict_species

def test_api_returns_string():
    input_data = {
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181,
        "body_mass_g": 3750,
        "sex": "Male",
        "island": "Torgersen",
        "year": 2007
    }

    pred = predict_species(input_data)

    assert isinstance(pred, str)

