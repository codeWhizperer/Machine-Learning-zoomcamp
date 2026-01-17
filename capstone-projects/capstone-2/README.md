# Trader Classification API

This project is a **machine learning-based API** for classifying traders as **Good (1)** or **Bad (0)** based on historical transaction activity. It uses an **XGBoost model** trained on trader features and exposes a **Flask REST API** for predictions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Data](#data)
- [Model Training](#model-training)
- [Model](#model)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [Using the API](#using-the-api)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The **Trader Classification ML project** collates transaction data of **5,000 traders** including the following columns:

- `tx_count_365d`
- `total_volume`
- `active_weeks`
- `avg_tx_value`
- `trader_activity_status`
- `trader_volume_status`
- `trader_consistency_status`

The target variable is **Good Trader (1)** or **Bad Trader (0)**.  

The aim of this project is to **train a machine learning model** to forecast a trader's potential category based on their transaction data.  

To ensure proper data balance and improve model robustness, **feature engineering and logarithmic transformations** were applied.

During model training, several algorithms were tested, including **Linear Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**, with **XGBoost outperforming all other models**.

---

## Features

The model uses the following features:

- `tx_count_365d`: Number of transactions in the last 365 days  
- `total_volume`: Total transaction volume  
- `active_weeks`: Number of weeks the trader was active  
- `avg_tx_value`: Average transaction value  
- `trader_activity_status`: Categorical – e.g., "Occasional User", "Frequent User"  
- `trader_volume_status`: Categorical – e.g., "Low Value", "High Value"  
- `trader_consistency_status`: Categorical – e.g., "Consistent", "Inconsistent"  

Additional engineered features:

- Log-transformed: `log_tx_count`, `log_total_volume`, `log_avg_tx_value`, `log_active_weeks`  
- Ratios: `volume_per_tx`, `volume_per_week`, `avg_weekly_tx`, `activity_intensity`, `tx_efficiency`  
- Interactions: `tx_volume_interaction`, `weeks_tx_interaction`, `consistency_volume`  
- Polynomial: `tx_count_squared`, `total_volume_squared`, `active_weeks_squared`  

---

## Data

The dataset contains **historical trader transactions** labeled as:

- `1` → Good Trader  
- `0` → Bad Trader  

It is split into **training**, **validation**, and **test** sets.

---

## Model Training

Multiple algorithms were tested:

### Linear Regression

```bash
ROC AUC: 0.97072
Precision/Recall/F1 (Validation):
0: 0.94 / 0.92 / 0.93
1: 0.92 / 0.94 / 0.93
Accuracy: 0.93
```

### Decision Tree

```bash
Validation Accuracy: 0.93
Validation ROC AUC: 0.958
Test Accuracy: 0.914
Test ROC AUC: 0.942
```


### Random Forest Classifier

```bash
Validation Accuracy: 0.937
Validation ROC AUC: 0.980
Test Accuracy: 0.917
Test ROC AUC: 0.977
```

### XGBOOST

Optimal threshold based on F1-score: 0.733

Validation:
Accuracy: 0.94
ROC AUC: 0.978
Precision/Recall/F1:
0: 0.94 / 0.94 / 0.94
1: 0.94 / 0.94 / 0.94
Confusion Matrix: [[472 28] [32 468]]

Test:
Accuracy: 0.912
ROC AUC: 0.977
Precision/Recall/F1:
0: 0.90 / 0.93 / 0.91
1: 0.92 / 0.90 / 0.91



**XGBoost** was selected as the final model due to superior performance.

---



## Model

- **Algorithm**: XGBoost Classifier  
- **Hyperparameters**:
  - `n_estimators=500`
  - `max_depth=5`
  - `learning_rate=0.05`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
  - `random_state=1`
- **Threshold Calibration**: Optimal threshold for F1-score (`0.733`) is used instead of 0.5 to balance precision and recall.  

### Model Performance Comparison

| Model | Validation Accuracy | Validation ROC AUC | Test Accuracy | Test ROC AUC | Notes |
|-------|-------------------|------------------|---------------|-------------|-------|
| Linear Regression | 0.93 | 0.9707 | 0.93 | 0.9707 | Good baseline |
| Decision Tree | 0.93 | 0.9579 | 0.914 | 0.9417 | Slight overfitting |
| Random Forest | 0.937 | 0.9797 | 0.917 | 0.9771 | Stable and robust |
| **XGBoost** | **0.94** | **0.9783** | 0.912 | 0.9766 | Best overall performance |




## Installation

1. Clone the repository:

```bash
git clone https://github.com/codeWhizperer/Machine-Learning-zoomcamp.git
cd Machine-Learning-zoomcamp/capstone-2
```

2. Create a Python environment using Pipenv:

```bash
pipenv install flask numpy pandas scikit-learn xgboost
pipenv shell
```


### Running the API

```bash
python predict.py

http://0.0.0.0:9696
```

### Using the API

```bash
Send a POST request to /predict with JSON data containing trader features. Example:

{
    "tx_count_365d": 10,
    "total_volume": 50.2,
    "active_weeks": 8,
    "avg_tx_value": 5.02,
    "trader_activity_status": "Regular User",
    "trader_volume_status": "Medium Value",
    "trader_consistency_status": "Consistent"
}

Response:

{
    "predicted_proba": 0.9913,
    "predicted_target": 1
}
```


## Contributing

This is an experimental and learning project. Please do not use for production purposes. Feel free to leave comments, suggest improvements, or contribute to the code.


## LICENSE

MIT License

