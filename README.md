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

1. **Exploratory Analysis** — Clean, normalize, and explore data using EDA, PCA, and clustering techniques.
2. **Predictive Modeling** — Train and evaluate an ensemble of machine learning models including XGBoost, Random Forest and Neural Networks to predict whether a patient will be a no-show.

## 🔧 Technical Approach

### Modeling Pipeline

- Load data from provided csv files in full batches or incrementally.
- Apply **Principal Component Analysis (PCA)** to reduce dimensionality of dense SDOH features.
- Train an ensemble of supervised models:
  - **Random Forest** — for feature importance reporting
  - **XGBoost** — for flexibility and handling class imbalance
  - **Neural Networks** — as an additional benchmark
- Perform hyperparameter tuning via grid search.
- Evaluate models using precision, ROC-AUC, and binary cross-entropy loss.


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

The primary metric is **precision** on the no-show class — we want to minimize false positives (predicting a no-show when the patient actually shows up), as this could result in legal and ethical consequences for the hospital.

### Target Confusion Matrix

|  | Actual No-Show | Actual Show |
|--|----------------|-------------|
| **Predicted No-Show** | > 0% | ~0% |
| **Predicted Show** | < 12% | ~88% |

> The current hospital baseline is to predict all patients will show up, resulting in a 12% false negative rate. Our model aims to match or improve on this while maintaining near-zero false positive predictions.

### Evaluation Strategy

- **5-Fold Cross Validation** on the training data (no separate evaluation set provided)
- **Primary Metric:** Precision on the no-show class
- **Secondary Metrics:** ROC-AUC, Binary Cross-Entropy Loss

### Fairness & Bias

We will report feature importances for each model to identify any social biases and share these findings with the CHOC team to ensure ethical model deployment.

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| **Database** | ClickHouse (locally hosted OLAP columnar DB) |
| **Data Processing** | Python, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Statistical Analysis** | R |
| **Machine Learning** | Scikit-Learn, XGBoost |
| **Dimensionality Reduction** | PCA (Scikit-Learn) |
| **Dashboard (Optional)** | Streamlit |
| **Version Control** | Git, GitHub |

---

## 📁 Project Structure

Billy

---

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
