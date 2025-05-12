## Introduction

nirdizati-light is a Python library for predictive process mining that focuses on the following aspects:

- trace encoding
- model training
- hyperparameter optimization
- model evaluation
- explainability

## Test
Run run_simple_pipeline.py


## Features

### Encodings

- Simple encoding
- Simple trace encoding
- Frequency encoding
- Complex encoding
- Loreley encoding
- Loreley complex encoding

### Labeling types

- Next activity (classification)
- Attribute string, i.e. outcome (classification)
- Remaining time (regression)
- Duration (regression)

### Predictive models

Classification:

- Random forest (scikit-learn)
- Decision tree (scikit-learn)
- KNN (scikit-learn)
- XGBoost (scikit-learn)
- SGD (scikit-learn)
- SVC (scikit-learn)
- LSTM (PyTorch)
- CustomPytorch, i.e. specify your own custom PyTorch model (PyTorch)

Regression:

- Random forest (scikit-learn)

### Hyperparameter optimization targets

- F1 score (classification)
- AUC (classification)
- Accuracy (classification)
- MAE (regression)

### Explainers

- ICE
- SHAP
- DiCE