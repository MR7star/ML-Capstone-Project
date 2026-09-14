```markdown
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

| Algorithm | R² Score | RMSE | MAE | 5-Fold CV R² | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Linear Regression | - | - | - | - | - |
| Ridge Regression | - | - | - | - | - |
| Lasso Regression | - | - | - | - | - |
| ElasticNet | - | - | - | - | - |
| Polynomial Regression | - | - | - | - | - |
| Decision Tree Regressor | - | - | - | - | - |
| Random Forest Regressor | - | - | - | - | - |
| Gradient Boosting Regressor | - | - | - | - | - |
| Support Vector Regressor | - | - | - | - | - |
| K-Nearest Neighbors Regressor | - | - | - | - | - |

*(Note: Metric values are automatically updated upon executing full grid search runs in `notebooks/regression.ipynb`.)*

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

```

```