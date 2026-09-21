# 23CSE301 Machine Learning Capstone Project

## Online Retail Sales Forecasting & Multi-Track Customer Analytics
**Academic Year:** 2026–27  
**Course:** 23CSE301 Machine Learning  
**Institution:** Amrita Vishwa Vidyapeetham  

---

## Team Members

* **Jishnu Nambiar** (CB.SC.U4CSE24019)
* **Muthu Rupesh M J** (CB.SC.U4CSE24030)
* **Ojas Joshi** (CB.SC.U4CSE24034)

---

## Problem Statement

The objective of this capstone project is to develop a complete, end-to-end Machine Learning pipeline analyzing global e-commerce transaction data. The project spans three comprehensive modeling tracks:
1. **Regression Track:** Predict the daily sales volume (`daily_sales_volume`) at the storefront-country level to assist in store-level inventory allocation and demand forecasting.
2. **Classification Track:** Classify transaction/order demand tiers across multiple machine learning baseline and ensemble models.
3. **Clustering Track:** Perform unsupervised segmentation on transaction behaviors and country-level purchasing patterns without relying on target labels during training.

---

## Dataset Description & Preprocessing Pipeline

The project utilizes the **Online Retail Dataset** containing international e-commerce transaction records. 

### Dataset Versions & Transformation Workflow

To perform analysis across both micro-level transaction trends and macro-level daily sales demand, the data pipeline generates two distinct structural versions of the dataset:

* **Version 1: Raw Transaction / Line-Item Dataset (`data/raw/data.csv`)**
  * **Dimensions:** 541,909 entries across 8 attributes.
  * **Original Features:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.
  * **Data Audit & Quality Control:** Initial audit identified 1,454 missing product descriptions, 135,080 missing customer IDs, and 5,268 duplicate transaction rows. Filtering out cancellations, returns, and invalid pricing (`Quantity > 0` and `UnitPrice > 0`) removed 11,805 invalid rows, leaving 530,104 clean transaction records.
  * **Line-Item Feature Engineering:** Formulated line-item revenue via `TotalSales = Quantity * UnitPrice` and parsed `InvoiceDate` into standard date formats.

* **Version 2: Storefront-Day Aggregated Dataset (`data/raw/data_1.csv`)**
  * **Dimensions:** 1,554 storefront-day aggregated records across 7 features.
  * **Aggregation Strategy:** Clean line-item transactions were grouped by `Date` and `Country` to construct a daily storefront summary dataset suitable for time-series and regression modeling.
  * **Aggregated Features:**
    * `Date`: Derived calendar date from `InvoiceDate`
    * `Country`: Geographic origin of the order
    * `product_variation_count`: Distinct product types sold (`StockCode` nunique)
    * `total_items_sold`: Total quantity of units sold (`Quantity` sum)
    * `total_orders`: Total unique invoices generated (`InvoiceNo` nunique)
    * `promotional_banner_count`: Derived promotional visibility index
    * `daily_sales_volume`: Target variable representing total daily store revenue (`TotalSales` sum)

---

## Project Tracks & Required Algorithms

The pipeline trains and evaluates 22 algorithms across the three capstone tracks:

### 1. Regression Track (10 Models)
Evaluated on the held-out test split of `data_1.csv` using R² score, RMSE, MAE, and 5-fold cross-validated R² for top performers:
* **Linear Regression:** Baseline parametric model for coefficient interpretability.
* **Ridge Regression:** L2 regularization with hyperparameter tuning on alpha.
* **Lasso Regression:** L1 regularization for feature selection and sparsity observation.
* **ElasticNet:** Combined L1 + L2 regularization tuning the l1_ratio.
* **Polynomial Regression:** Feature expansion using `PolynomialFeatures` prior to linear fitting.
* **Decision Tree Regressor:** Non-linear decision boundaries with `max_depth` tuning and feature importance analysis.
* **Random Forest Regressor:** Ensemble averaging baseline tuning `n_estimators`.
* **Gradient Boosting Regressor:** Sequential boosting ensemble tuning learning rates and estimators.
* **Support Vector Regressor (SVR):** Feature-scaled kernelized regression tuning C and kernel functions.
* **K-Nearest Neighbors Regressor:** Distance-based regression tuning k and scaling impacts.

### 2. Classification Track (10 Models)
Evaluated across Accuracy, Precision, Recall, Weighted F1-score, Confusion Matrix, and OvR ROC-AUC:
* **Part A (Review 1):** Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes, Decision Tree Classifier, Support Vector Machine (SVC).
* **Part B (Review 2):** Random Forest Classifier, AdaBoost Classifier, Gradient Boosting Classifier, Bagging Classifier, MLP Classifier.

### 3. Clustering Track (2 Models)
Unsupervised partitioning evaluated via Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, and 2D PCA / t-SNE projections:
* **K-Means Clustering:** Centroid-based clustering with Elbow curve analysis.
* **Agglomerative Hierarchical Clustering:** Hierarchical decomposition with dendrogram visualizations.

---

## Summary of Results

### Regression Track Performance Comparison

| Rank | Model | R² | RMSE | MAE |
|---:|---|---:|---:|---:|
| 1 | Support Vector Regressor | 0.8964 | 3861.66 | 1588.10 |
| 2 | Random Forest Regressor | 0.8786 | 4180.98 | 1564.38 |
| 3 | K-Nearest Neighbors Regressor | 0.8690 | 4344.02 | 1823.61 |
| 4 | Polynomial Regression | 0.8658 | 4395.71 | 1748.68 |
| 5 | Gradient Boosting Regressor | 0.8632 | 4438.79 | 1688.14 |
| 6 | Linear Regression | 0.8427 | 4758.54 | 1993.78 |
| 7 | Ridge Regression | 0.8427 | 4759.92 | 1992.16 |
| 8 | ElasticNet Regression | 0.8373 | 4840.66 | 1977.25 |
| 9 | Lasso Regression | 0.8340 | 4888.65 | 1983.81 |
| 10 | Decision Tree Regressor | 0.7711 | 5740.90 | 2071.07 |


### Classification Track Performance Comparison

Part A — Final Model Comparison

| Algorithm | Accuracy | Weighted F1 | Macro F1 | Positive Recall | Positive Precision | Positive F1 | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Decision Tree** | **89.74%** | **0.8931** | **0.7881** | **57.9%** | **70.6%** | **0.6360** | **0.9177** | **0.6882** |
| Support Vector Machine | 88.89% | 0.8811 | 0.7591 | 50.0% | 69.7% | 0.5823 | 0.8662 | 0.6646 |
| Logistic Regression | 88.12% | 0.8630 | 0.7072 | 35.6% | 74.3% | 0.4814 | 0.8875 | 0.6237 |
| K-Nearest Neighbors | 87.19% | 0.8450 | 0.6592 | 26.4% | 74.3% | 0.3900 | 0.8301 | 0.5318 |
| Gaussian Naive Bayes | 68.78% | 0.7301 | 0.6088 | 77.0% | 30.1% | 0.4330 | 0.7941 | 0.4475 |

---

## Repository Structure

```text
ML-Capstone-Project/
├── README.md                  # Project overview, guidelines compliance, and setup guide
├── requirements.txt            # Python dependencies with version specifications
├── data/                      # Data storage directory
│   ├── raw/
│   │   ├── data.csv           # Version 1: Line-item raw online retail dataset
│   │   └── data_1.csv         # Version 2: Storefront-day aggregated regression dataset
│   └── processed/             # Cleaned, scaled, and split dataset artifacts
├── notebooks/                 # Model development notebooks
│   ├── regression.ipynb       # Data audit, versioning, EDA, and 10 regression algorithms
│   ├── classification.ipynb   # 10 classification algorithms (Part A & Part B)
│   └── clustering.ipynb       # K-Means, Hierarchical clustering, PCA & t-SNE visualisations
├── models/                    # Serialized model artifacts (.pkl files via joblib)
└── app/                       # Interactive web interface / GUI deployment code

```

---

## Instructions to Setup & Run

### Environment Prerequisites

* Python 3.10+
* Virtual Environment (`venv` or `conda`)

### 1. Installation & Environment Setup

Clone the repository and install required packages:

```bash
git clone [https://github.com/muthurupesh/ML-Capstone-Project.git](https://github.com/muthurupesh/ML-Capstone-Project.git)
cd ML-Capstone-Project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Execution Order

Run the notebooks top-to-bottom to reproduce data preprocessing and model evaluation:

```bash
jupyter notebook

```

1. **`notebooks/regression.ipynb`**: Performs dataset audit on `data.csv`, filters invalid records, generates `data_1.csv`, executes EDA, and trains all 10 regression algorithms.
2. **`notebooks/classification.ipynb`**: Trains 10 classification algorithms across Part A and Part B, generating confusion matrices and ROC-AUC metrics.
3. **`notebooks/clustering.ipynb`**: Fits K-Means and Agglomerative Clustering, generating Elbow curves, Dendrograms, and PCA/t-SNE cluster plots.

### 3. Web Interface / Bonus Deployment (Optional)

Launch the interactive web GUI:

```bash
streamlit run app/main.py

```

---

## Academic Integrity & Generative AI Disclosure

* All analytical interpretations, EDA insights, and feature engineering strategies were designed and implemented by the team members.
* Generative AI tools were utilized strictly for code scaffolding, layout structuring, and documentation formatting assistance.

