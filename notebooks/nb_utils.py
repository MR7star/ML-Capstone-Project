"""
nb_utils.py — Shared Utilities for ML Capstone Classification Track
Includes:
- Visual UI helpers: title(), hero(), finish()
- Data pipeline: load_and_preprocess_data() ensuring zero data leakage,
  stratified 80:20 split, engineered features, and proper scaling/encoding.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from IPython.display import display, HTML

# Visual Color Tokens
ACCENT = "#2563eb"       # Royal blue
SUCCESS = "#16a34a"      # Forest green
WARNING = "#d97706"      # Amber
MUTED = "#64748b"        # Slate muted
DARK = "#0f172a"         # Deep slate


def title(text: str = "", kicker: str = None):
    """
    Optional section title helper. Kept simple and theme-safe.
    """
    pass


def hero(cards: list):
    """
    Prints a clean, single-line summary of key metrics.
    Works naturally in both light and dark themes.
    cards: list of tuples -> (stat_value, label, *optional_color)
    """
    metrics = " | ".join([f"{label}: {stat}" for stat, label, *_ in cards])
    print(metrics)


def finish(fig, title_text: str, note: str = None):
    """
    Standardizes plot layout, title, and optional note.
    """
    fig.suptitle(title_text, fontsize=12, fontweight="bold", y=1.02)
    if note:
        fig.text(0.5, -0.04, note, ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    plt.show()


def get_dataset_path() -> Path:
    """Finds the dataset path reliably from notebooks/ or root."""
    possible = [
        Path("../data/raw/online_shoppers_intention.csv"),
        Path("data/raw/online_shoppers_intention.csv"),
        Path("a:/ml_capstone/data/raw/online_shoppers_intention.csv")
    ]
    for p in possible:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("Could not locate online_shoppers_intention.csv in data/raw/")


def load_raw_data() -> pd.DataFrame:
    """Loads raw dataset without preprocessing for EDA."""
    path = get_dataset_path()
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates intuitive, justified engineered features:
    1. Total_Pages_Visited = Administrative + Informational + ProductRelated
    2. Total_Duration = Administrative_Duration + Informational_Duration + ProductRelated_Duration
    3. Avg_Duration_Per_Page = Total_Duration / (Total_Pages_Visited + 1e-5)
    4. Exit_to_Bounce_Ratio = ExitRates / (BounceRates + 1e-5)
    """
    data = df.copy()
    data["Total_Pages_Visited"] = data["Administrative"] + data["Informational"] + data["ProductRelated"]
    data["Total_Duration"] = data["Administrative_Duration"] + data["Informational_Duration"] + data["ProductRelated_Duration"]
    data["Avg_Duration_Per_Page"] = data["Total_Duration"] / (data["Total_Pages_Visited"] + 1e-5)
    data["Exit_to_Bounce_Ratio"] = data["ExitRates"] / (data["BounceRates"] + 1e-5)
    return data


def load_and_preprocess_data(test_size: float = 0.20, random_state: int = 42, drop_duplicates: bool = False):
    """
    Loads, cleans, engineers features, and splits the data.
    Guarantees:
    - Stratified split preserving class balance (~15.5% True, 84.5% False)
    - Zero data leakage: Scaler & OneHotEncoder are fitted ONLY on X_train.
    - Scaled representations available with feature names aligned.
    """
    raw_df = load_raw_data()
    if drop_duplicates:  # only used for the duplicate sensitivity check
        raw_df = raw_df.drop_duplicates()
    df = engineer_features(raw_df)

    # Encode target Revenue (True -> 1, False -> 0)
    y = df["Revenue"].astype(int)
    X = df.drop(columns=["Revenue"])

    # Identification of columns
    num_cols = [
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay",
        "OperatingSystems", "Browser", "Region", "TrafficType",
        "Total_Pages_Visited", "Total_Duration", "Avg_Duration_Per_Page", "Exit_to_Bounce_Ratio"
    ]
    cat_cols = ["Month", "VisitorType", "Weekend"]

    # Stratified Train-Test Split (fitted on train only)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Preprocessing Pipeline
    scaler = StandardScaler()
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

    # Fit strictly on train!
    X_train_num = scaler.fit_transform(X_train_raw[num_cols])
    X_test_num = scaler.transform(X_test_raw[num_cols])

    X_train_cat = encoder.fit_transform(X_train_raw[cat_cols])
    X_test_cat = encoder.transform(X_test_raw[cat_cols])

    cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names

    X_train_scaled = np.hstack([X_train_num, X_train_cat])
    X_test_scaled = np.hstack([X_test_num, X_test_cat])

    X_train_df = pd.DataFrame(X_train_scaled, columns=all_feature_names, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=all_feature_names, index=X_test_raw.index)

    return {
        "X_train": X_train_df,
        "X_test": X_test_df,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "feature_names": all_feature_names,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "scaler": scaler,
        "encoder": encoder
    }


def get_scores(model, X):
    """Positive-class probability, or decision_function for models without predict_proba (SVC)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate_classifier(name, model, X_test, y_test, fit_seconds=np.nan):
    """Returns one row of test-set metrics for a fitted binary classifier."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 roc_auc_score, average_precision_score, confusion_matrix)
    y_pred = model.predict(X_test)
    scores = get_scores(model, X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_test, y_pred, average="weighted"),
        "F1 (weighted)": f1_score(y_test, y_pred, average="weighted"),
        "F1 (macro)": f1_score(y_test, y_pred, average="macro"),
        "Precision (buy)": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall (buy)": recall_score(y_test, y_pred, pos_label=1),
        "F1 (buy)": f1_score(y_test, y_pred, pos_label=1),
        "ROC-AUC": roc_auc_score(y_test, scores),
        "PR-AUC": average_precision_score(y_test, scores),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Fit time (s)": fit_seconds,
    }


def plot_confusion(ax, y_true, y_pred, title):
    """Confusion matrix heatmap with the same labels and colours used across the notebook."""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticklabels(["No Purchase", "Purchase"])
    ax.set_yticklabels(["No Purchase", "Purchase"], rotation=0)
    return cm


def style_table(df, highlight_cols=(), lower_better=(), fmt="{:.4f}"):
    """Formats numeric columns and highlights the best value in each listed column."""
    num_cols = df.select_dtypes("number").columns
    int_cols = [c for c in num_cols if pd.api.types.is_integer_dtype(df[c])]
    styler = df.style.format({c: ("{:,}" if c in int_cols else fmt) for c in num_cols})
    hi = [c for c in highlight_cols if c not in lower_better]
    lo = [c for c in highlight_cols if c in lower_better]
    if hi:
        styler = styler.highlight_max(subset=hi, color="#bfdbfe")
    if lo:
        styler = styler.highlight_min(subset=lo, color="#bfdbfe")
    return styler.hide(axis="index")


def part_a_models(random_state: int = 42):
    """The five Part A classifiers with the configurations selected in Review 1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=13, metric="manhattan"),
        "Gaussian Naive Bayes": GaussianNB(var_smoothing=1e-3),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=random_state),
        "Support Vector Machine": SVC(kernel="rbf", C=5.0, random_state=random_state),
    }
