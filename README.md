# Model Performance & Comparison: Adult Income Prediction

This repository contains the end-to-end data cleaning, preprocessing, and machine learning model benchmarking for predicting whether an individual's annual income exceeds $50,000 using the **Adult Census Income dataset** (UCI Machine Learning Repository).

---

## 1. Project Overview

- **Target Variable:** Income level (`<=50K` [0] vs. `>50K` [1])
- **Dataset:** Adult Census Income Dataset
- **Preprocessing Pipeline:** Data cleaning, handling missing and categorical values, train-test split (80/20 with `random_state=42`), and feature standardization using `StandardScaler`.
- **Optimization Strategy:** 5-fold cross-validation with `RandomizedSearchCV` across 8 classification algorithms.

---

## 2. Model Performance Comparison

Hyperparameter tuning was conducted using `RandomizedSearchCV` (5-fold cross-validation) across 8 classification algorithms.

| Model | Best Hyperparameters | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Top Overall Performer)** | `n_estimators: 200`, `max_depth: 5`, `learning_rate: 0.1` | **86.67%** | **76.57%** | **63.93%** | **69.68%** | **78.88%** |
| **Gradient Boosting** | `n_estimators: 200`, `max_depth: 5`, `learning_rate: 0.1` | 86.44% | 76.04% | 63.40% | 69.15% | 78.55% |
| **Random Forest** | `n_estimators: 50`, `max_depth: 15`, `min_samples_split: 2`, `min_samples_leaf: 2` | 86.21% | 77.80% | 59.41% | 67.38% | 77.04% |
| **Decision Tree** | `max_depth: 10`, `min_samples_split: 10`, `min_samples_leaf: 1`, `criterion: 'gini'` | 84.74% | 73.34% | 57.03% | 64.16% | 75.25% |
| **AdaBoost** *(Highest Precision)* | `n_estimators: 200`, `learning_rate: 0.1` | 84.72% | **80.70%** | 47.62% | 59.90% | 72.02% |
| **Support Vector Classifier (SVC)** | `C: 1`, `kernel: 'rbf'`, `gamma: 'scale'` | 84.09% | 72.77% | 53.69% | 61.79% | 73.68% |
| **Logistic Regression** | `C: 0.01`, `penalty: 'l1'`, `solver: 'saga'` | 83.49% | 71.76% | 51.26% | 59.80% | 72.45% |
| **K-Nearest Neighbors (KNN)** | `n_neighbors: 13`, `weights: 'uniform'`, `algorithm: 'ball_tree'` | 82.75% | 65.73% | 58.55% | 61.93% | 74.47% |

---

## 3. Key Performance Insights

- **Top Performer (XGBoost):** XGBoost achieved the highest overall predictive performance across **Accuracy (86.67%)**, **F1 Score (69.68%)**, and **ROC AUC (78.88%)**, demonstrating superior handling of tabular census features.
- **Ensemble Superiority:** Tree-based ensemble models (XGBoost, Gradient Boosting, and Random Forest) consistently outperformed linear and distance-based baselines, all exceeding **86% accuracy**.
- **High-Precision Alternative (AdaBoost):** AdaBoost achieved the highest precision (**80.70%**), making it the most conservative model for minimizing false positives when predicting high earners (`>50K`).
- **Linear & Instance-based Baselines:** Logistic Regression (**83.49%**) and SVC (**84.09%**) provided strong baselines post-standardization, though limited in capturing non-linear interactions compared to gradient boosted decision trees.
- **ROC Discrimination:** Both XGBoost and Gradient Boosting achieved strong discriminatory power with ROC AUC scores near **0.79**, effectively balancing the class imbalance present in the adult income distribution.

---

## 4. Repository Structure

```text
adult_income/
├── datasets/
│   └── adult/
│       ├── adult.data
│       ├── adult.names
│       └── adult.test
├── notebooks/
│   ├── adult.data
│   ├── adult_cleaned.csv
│   ├── data_cleaning.ipynb
│   └── model_training.ipynb
└── README.md
```
