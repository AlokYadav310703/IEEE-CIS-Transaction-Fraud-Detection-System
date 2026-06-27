# IEEE-CIS Fraud Detection

A complete machine learning pipeline for the **IEEE-CIS Fraud Detection** Kaggle competition. The objective is to predict whether an online transaction is fraudulent using transaction and identity information provided by **Vesta Corporation**.

---

# Project Structure

```text
├── data/
│   └── raw/
│       ├── train_transaction.csv
│       ├── train_identity.csv
│       ├── test_transaction.csv
│       └── test_identity.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_feature_engineering___model_training.ipynb
└── README.md
```

---

# Dataset Overview

The dataset consists of two paired files linked using **TransactionID**.

| File                    | Description                                      |
| ----------------------- | ------------------------------------------------ |
| `train_transaction.csv` | Transaction details with fraud label (`isFraud`) |
| `train_identity.csv`    | Device, browser, and identity information        |
| `test_transaction.csv`  | Transaction details without labels               |
| `test_identity.csv`     | Identity information without labels              |

## Dataset Characteristics

* ~590,000 training transactions
* ~500,000 test transactions
* Fraud rate of approximately **3.5%** (highly imbalanced)
* Over **400 features**
* Numerous anonymized variables (`V1–V339`, `C1–C14`, `D1–D15`, `M1–M9`, `id_01–id_38`)
* Heavy missing values (many columns contain **70–95%** null values)

---

# Notebooks

## 1. `01_eda.ipynb` — Exploratory Data Analysis

This notebook focuses on understanding the dataset and assessing data quality.

### Contents

* Column group inventory (`V`, `C`, `D`, `M`, `card`, `id`)
* Memory optimization using dtype downcasting
* Fraud rate analysis and class imbalance visualization
* Transaction amount distribution
* Log-transformation diagnostics
* Missing value profiling
* Chi-squared feature analysis
* Univariate feature importance using AUC

---

## 2. `02_feature_engineering___model_training.ipynb` — Full ML Pipeline

An end-to-end machine learning pipeline from raw data preprocessing to model ensembling and evaluation.

---

# Preprocessing

* Merge transaction and identity datasets using `TransactionID`
* Remove columns with more than **90% missing values**
* Remove near-zero variance features using `VarianceThreshold`
* Compress high-dimensional `V` features using PCA grouped by missingness pattern

---

# Feature Engineering

### Transaction Features

* Log transformation of `TransactionAmt`
* Decimal component extraction from transaction amount
* Amount-to-card historical mean ratio
* Transaction amount percentile within each card group

### Time Features

* Hour of day
* Day of week
* Week number
* Hour × Day interaction feature

### User Features

* Pseudo User ID:

  * `uid = card1 + addr1 + email`
  * `uid2 = card1 + card2`

### Frequency Encoding

Applied to:

* Card features
* Email domains
* Device information

### Card Aggregations

Per-card statistics including:

* Mean transaction amount
* Standard deviation of transaction amount
* Mean and standard deviation of D-features

### Transaction Velocity

* Transactions per card
* Transactions per card pair

### D-Feature Engineering

* Card-wise normalization
* Missing-value indicators

### Identity Features

* Email domain match flag
* Big email provider indicator
* Identity richness score

### Missing Value Features

* Global null count per row

---

# Class Imbalance Handling

The dataset is highly imbalanced (~3.5% fraud).

To improve recall, **SMOTE** oversampling is applied to the training data, increasing the fraud ratio to approximately **15%**.

---

# Models Trained

| Model                        | Configuration                                                              |
| ---------------------------- | -------------------------------------------------------------------------- |
| Logistic Regression          | Baseline (`class_weight='balanced'`)                                       |
| LightGBM GBDT                | Optuna hyperparameter tuning                                               |
| LightGBM DART                | Dropout-based boosting                                                     |
| XGBoost                      | Optuna tuning                                                              |
| CatBoost                     | Optuna tuning with native categorical handling                             |
| Multi-Layer Perceptron (MLP) | PyTorch implementation using Focal Loss and Cosine Learning Rate Scheduler |

---

# Ensemble Strategy

Predictions from multiple models are combined using:

* Rank Averaging
* Weighted Rank Averaging

Typical ensemble:

```
LightGBM (GBDT)
+ LightGBM (DART)
+ XGBoost
+ CatBoost
+ MLP Neural Network (weighted ×0.5)
```

The final ensemble is selected based on the highest **Validation PR-AUC**.

---

# Evaluation

## Primary Metric

* **PR-AUC (Average Precision)**

Since only **3.5%** of transactions are fraudulent, PR-AUC is more informative than ROC-AUC.

## Secondary Metric

* ROC-AUC

## Threshold Optimization

The classification threshold is selected by maximizing the **F2 Score**, placing greater emphasis on recall.

## Generated Evaluation Plots

* ROC Curve
* Precision–Recall Curve
* Confusion Matrix
* Feature Importance

---

# Installation

Install the required packages:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn \
            lightgbm xgboost catboost optuna shap \
            imbalanced-learn torch torchvision openpyxl
```

GPU acceleration is automatically used for **LightGBM**, **XGBoost**, and **PyTorch** whenever available.

---

# Usage

1. Download the IEEE-CIS Fraud Detection dataset from Kaggle.
2. Place the CSV files inside:

```text
data/raw/
```

3. Run the exploratory analysis notebook:

```text
01_eda.ipynb
```

4. Run the complete training pipeline:

```text
02_feature_engineering___model_training.ipynb
```

5. Generate predictions and create the submission file.

> **Note:**
> The notebook paths are configured for the Kaggle environment (`/kaggle/input/`). If running locally, update these paths to point to your local `data/raw/` directory.

---

# Key Design Decisions

## Why PR-AUC?

With only **3.5%** fraudulent transactions, ROC-AUC can produce overly optimistic scores. PR-AUC focuses specifically on the minority class, making it a better evaluation metric for fraud detection.

---

## Time-Based Validation

Instead of using a random train-validation split, the last **20%** of transactions sorted by `TransactionDT` are used as the validation set. This better reflects real-world deployment where future transactions must be predicted from historical data.

---

## V-Feature PCA Compression

The dataset contains **339 anonymized V-features**. These are grouped by similar missing-value patterns and compressed using PCA to:

* Reduce dimensionality
* Lower memory usage
* Remove redundant information
* Preserve most of the original variance

---

## Rank-Average Ensembling

Different models often produce probability scores on different scales. Rank averaging converts predictions into ranks before combining them, making the ensemble more robust and less sensitive to calibration differences between models.

---

# Results

The complete pipeline integrates:

* Advanced feature engineering
* PCA-based dimensionality reduction
* SMOTE for class imbalance
* Hyperparameter optimization with Optuna
* Multiple gradient boosting models
* Neural network training with Focal Loss
* Robust rank-based model ensembling

This approach provides a strong baseline for the IEEE-CIS Fraud Detection competition while maintaining reproducibility and scalability.
