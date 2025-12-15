from penguin_lib.models.trainer import PenguinTrainer

def predict_species(input_dict):
    """
    input_dict must contain:
    - bill_length_mm
    - bill_depth_mm
    - flipper_length_mm
    - body_mass_g
    - sex
    - island
    - year
    """

    trainer = PenguinTrainer()
    X, y = trainer.load_and_prepare("data/penguins.csv")
    results = trainer.train(X, y)

    model = trainer.best_model
    preprocessor = trainer.preprocessor
    feature_creator = trainer.feature_creator

    # Turn input into DataFrame
    import pandas as pd
    df = pd.DataFrame([input_dict])

    # Create same engineered features
    df = feature_creator.create_all(df)

    # Preprocess input
    X_processed = preprocessor.transform(df)

    # Predict
    pred = model.predict(X_processed)[0]

    return pred
