from penguin_lib.api.predict import predict_species

sample = {
    "bill_length_mm": 40.0,
    "bill_depth_mm": 18.0,
    "flipper_length_mm": 195.0,
    "body_mass_g": 4000.0,
    "sex": "male",
    "island": "Biscoe",
    "year": 2008
}

print("Predicted species:", predict_species(sample))
