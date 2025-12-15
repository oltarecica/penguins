# penguins
Library + API for predicting penguin species (Computing for Data Science final project)

## how to extend the library

---

## add a new preprocessor
All preprocessors live inside:
`penguin_lib/preprocessing/`

To add a new one:
1. Create a new file (for example `my_preprocessor.py`)
2. Implement a class with `fit`, `transform`, and `fit_transform`
3. Import and use it inside the trainer if needed

---

## add a new feature
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

### add a new model

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

## add a new metric
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

## how to run the project

### create the environment
```bash
uv sync
```

### run the training pipeline
```bash
uv run python test_training.py
```

### run the evaluation
```bash
uv run python test_evaluation.py
```

### run all tests
```bash
uv run pytest
```

### project layout
```text
penguin_lib/            library code
  preprocessing/        preprocessing logic
  features/             feature engineering
  models/               model training
  metrics/              evaluation metrics
tests/                  unit tests
data/                   dataset
test_pipeline.py        checks preprocessing + features
test_training.py        trains the model
test_evaluation.py      runs metrics