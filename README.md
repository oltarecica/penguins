# Penguins
Library + API for predicting penguin species (Computing for Data Science final project)

## how to extend the library

---

## Add a new preprocessor
All preprocessors live inside:
`penguin_lib/preprocessing/`

To add a new one:
1. Create a new file (for example `my_preprocessor.py`)
2. Implement a class with `fit`, `transform`, and `fit_transform`
3. Import and use it inside the trainer if needed

---

## Add a new feature
All feature engineering lives in:
`penguin_lib/features/feature_creator.py`

To add a new feature:
1. Open `feature_creator.py`
2. Define a new method (for example `create_new_feature`)
3. Add the feature computation inside the method
4. Register it inside `create_all()` so it is automatically applied

Example:
```python
def create_new_feature(self, df):
    df["new_feature"] = df["bill_length_mm"] / df["bill_depth_mm"]
    return df
```
---

### Add a new model

Models are defined in:
`penguin_lib/models/trainer.py`

To add a new model:
1. Import the model at the top of `trainer.py`
   (example: `from sklearn.svm import SVC`)
2. Replace the current model or allow choosing between models
3. Update the hyperparameter grid if needed

Example:
```python
from sklearn.svm import SVC

model = SVC(probability=True)
```
---

## Add a new metric
Metrics are defined in:
`penguin_lib/metrics/evaluation.py`

To add a new metric:
1. Import the metric from scikit-learn  
   (example: `from sklearn.metrics import roc_auc_score`)
2. Compute it inside the `evaluate()` method
3. Add it to the returned dictionary

Example:
```python
"auc": roc_auc_score(y_true, y_pred_proba, multi_class="ovr")
```
---
## API
The library provides a simple API function that predicts the penguin species for a single datapoint.

Location:
`penguin_lib/api/predict.py`

Function:
`predict_species(input_dict)`

Description:

The API:
-loads the penguins dataset
-trains the model using the full pipeline
-applies the same feature engineering and preprocessing steps
-returns the predicted species for one datapoint

Excpected input:
A dictionary with the same format as one row of the dataset.

Required fields:
bill_length_mm
bill_depth_mm
flipper_length_mm
body_mass_g
sex
island
year

Example:
```python
input_data = {
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750,
    "sex": "Male",
    "island": "Torgersen",
    "year": 2007
}
```
### How to run the API
From the project root:
```bash
uv run python test_api.py
```
The API returns a single predicted species label

## How to run the project

### Create the environment
```bash
uv sync
```

### Run the training pipeline
```bash
uv run python test_training.py
```

### Run the evaluation
```bash
uv run python test_evaluation.py
```

### Run all tests
```bash
uv run pytest
```

### Project layout
penguin_lib/                 core library
  api/                       prediction API
    predict.py               single-datapoint prediction
  preprocessing/             preprocessing logic
  features/                  feature engineering
  models/                    model training and selection
  metrics/                   evaluation metrics
  utils/                     shared utilities
tests/                       unit tests
data/                        dataset
test_pipeline.py             end-to-end pipeline test
test_training.py             model training script
test_evaluation.py           model evaluation script
test_api.py                  API usage example
