# 🏥 Predicting No-Shows to Appointments at CHOC

> A machine learning pipeline to predict patient appointment no-shows using Social Determinants of Health (SDOH) data, developed in partnership with the Children's Hospital of Orange County (CHOC).

---

## 👥 Team

| Name | GitHub |
|------|--------|
| Arun Malani | [@apmalani](https://github.com/apmalani) |
| Himal Malik | [@anonymous-himalayas](https://github.com/anonymous-himalayas) |
| Zac Jayachandran | [@Zeksauce](https://github.com/Zeksauce) |
| Billy Li | [@billy1734li](https://github.com/billy1734li) |

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Technical Approach](#technical-approach)
- [Datasets](#datasets)
- [Models & Evaluation](#models--evaluation)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

---

## 📌 Project Overview

Missed clinic appointments are a significant problem for both patients and healthcare institutions. They restrict access to care for patients and impose unnecessary financial burdens on hospitals. This project addresses the problem of appointment no-shows at CHOC by building a predictive model grounded in Social Determinants of Health (SDOH).

Our approach is two-fold:

1. **Exploratory Analysis** — Clean, normalize, and explore data using EDA and data engineering.
2. **Predictive Modeling** — Train and evaluate an ensemble of machine learning models including XGBoost, Random Forest and Logistic Regression to predict whether a patient will be a no-show.

## 🔧 Technical Approach

### Modeling Pipeline

- Load data from provided csv files in full batches or incrementally.
- Apply data cleaning, partitioning and engineering.
- Train an ensemble of supervised models:
  - **Random Forest** — for feature importance reporting
  - **XGBoost** — for flexibility and handling class imbalance
  - **Logistic Regression** — as an additional benchmark
- Perform hyperparameter tuning via grid search.
- Evaluate models using precision/recall, ROC-AUC and F1.


## 📊 Datasets

All three datasets are linked by FIPS code. CHOC data contains each patient's FIPS code, and SVI/COI contain FIPS-level SDOH metrics. FIPS codes were removed from the final dataset prior to delivery.

| Dataset | Source | Description |
|---------|--------|-------------|
| **CHOC Appointment Data (2022)** | Proprietary — CHOC | Patient appointment records including appointment day, time of day and no-show history. |
| **Social Vulnerability Index (SVI) 2022** | [CDC SDOH Data Catalog](https://www.atsdr.cdc.gov/place-health/php/svi/) | FIPS-level data covering Socioeconomic Status, Household Characteristics, Racial & Ethnic Status and Housing/Transportation. |
| **Child Opportunity Index (COI) 2010 and 2015** | [COI Census Tract Data](https://www.diversitydatakids.org/research-library/child-opportunity-index/child-opportunity-index-30-2023-census-tract-data) | FIPS-level data across Education, Health & Environment, and Social & Economic domains. Scores provided as z-scores, 5-part categories, and ratings at national, state, and metro levels. |

> ⚠️ **Note:** The CHOC dataset contains protected medical information and cannot be shared publicly. SDOH datasets are publicly available via the links above.

---

## 🤖 Models & Evaluation

### Target Metric

The primary metric is **precision** on the no-show class — we want to minimize false positives (predicting a no-show when the patient actually shows up), as this would waste hospital resources.

### Target Confusion Matrix

|  | Actual No-Show | Actual Show |
|--|----------------|-------------|
| **Predicted No-Show** | > 0% | ~0% |
| **Predicted Show** | < 12% | ~88% |

> The current hospital baseline is to predict all patients will show up, resulting in a perfect precision but 0 recall. Our model aims to match or improve on recall while maintaining high precision.

### Evaluation Strategy

- **5-Fold Cross Validation** on the training data (no separate evaluation set provided)
- **Primary Metric:** Precision on the no-show class
- **Secondary Metrics:** ROC-AUC, Binary Cross-Entropy Loss

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| **Data Processing** | Python, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Statistical Analysis** | R |
| **Machine Learning** | Scikit-Learn, XGBoost |
| **Version Control** | Git, GitHub |

---

## 📁 Project Structure

### Data
Contains all datasets and data-processing scripts.

> **Note:** Raw clinical datasets are excluded from this repository due to HIPAA/privacy requirements.

| File | Description |
|--------|-------------|
| `Data Dict.txt` | Data dictionary of all our data. |
| `data_partitioner.py` | Partitions datasets into 3 sets by which SDOH index is missing. |
| `fake_data.ipynb` | Creates a representative subsample for demonstrations without exposing protected patient information. |
| `initial_clean_exploration.ipynb` | Performs initial cleaning, validation, and exploration of the raw clinical data. |

---

### EDA
| File | Description |
|--------|-------------|
| `exploration.ipynb` | Our deep dive into the raw data and trying to see trends within the data between clinical data and SDOH indices data|
| `only_indexes.ipynb` | Our exploration in whether there is a correlation between index data and a child no-showing for their appointment |
| `other_factors.ipynb` | Exploring other factors not limited to SDOH data to see if there are other publicly available datasets that could help in our modeling|


---

### Feature Engineering
| File | Description |
|--------|-------------|
| `feature_engineering_pipe.ipynb` | Data pipeline that abstracts away sensitive EHR data that corresponds to a patient for easier processing and data handling |

---

### Logistic Regression
| File | Description |
|--------|-------------|
| `stepwise_logistic_regression.rmd` | This file was made to leverage the use of logistic regression for only CHOC Data. This implements 5 different models all originating from the baseline stepwise inplemented model: Quadratic Terms, Interaction Terms, Quasibinomial, Combo (Quadratic and Interaction Terms), and Weighted. Every model has a mapped ROC Curve that was plotted into one plot to show that there was no improvement from the orignal stepwise model. |

---

### Random Forest
| File | Description |
|--------|-------------|
| `initial_tree.ipynb` | Initial decision tree model development, including baseline training, hyperparameter exploration, and performance evaluation. |
| `RFECV_tree.ipynb` | Decision tree model trained and evaluated using Recursive Feature Elimination with Cross-Validation (RFECV) to identify the most informative features and improve model performance. |
---
### Submission
| File | Description |
|--------|-------------|
| `best_precision_xgb_model.pkl` | Trained XGBoost model optimized to maximize precision, minimizing false positive no-show predictions. |
| `best_recall_xgb_model.pkl` | Trained XGBoost model optimized to maximize recall, maximizing identification of potential no-show appointments. |
| `fake_choc_data.csv` | De-identified sample dataset used to demonstrate model predictions without exposing protected patient information. |
| `submission.ipynb` | Demonstrates the complete prediction pipeline, including loading the trained models, generating predictions on a testing dataset, and evaluating model performance. This notebook serves as an example of the deliverable that could be provided to CHOC. |

---
### XGBoost Model
| File | Description |
|--------|-------------|
| `XGBoost_model.ipynb` | Training and exploratory notebook for hyperparameter tuning and evaluating an array of XGBoost models|

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- R (for statistical analysis)

### Installation

```bash
# Clone the repository
git clone https://github.com/anonymous-himalayas/170A-CHOC.git
cd 170A-CHOC

# Install Python dependencies
pip install -r requirements.txt
```

### Sample
To run the sample of how our model would perform on a synthetic dataset:
- navigate to `submission` directory
- open `submission.ipynb` and connect to python kernel
- run all cells in the Jupyter Notebook


## 🤝 Contributing
This is an academic research project in partnership with CHOC. If you are a team member:

- Create a new branch for your feature: git checkout -b feature/your-feature-name
- Commit your changes: git commit -m "Add your feature"
- Push to the branch: git push origin feature/your-feature-name
- Open a Pull Request for review
## ⚖️ Privacy & Compliance
This project handles protected health information (PHI) from CHOC. All patient data is stored locally only and is never uploaded to any external server or repository. This project complies with HIPAA regulations.

🔒 Do NOT commit any patient data or CHOC proprietary data to this repository.

## 📄 License
This project is developed for academic and research purposes in collaboration with the Children's Hospital of Orange County (CHOC). All rights reserved.
