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


def load_and_preprocess_data(test_size: float = 0.20, random_state: int = 42):
    """
    Loads, cleans, engineers features, and splits the data.
    Guarantees:
    - Stratified split preserving class balance (~15.5% True, 84.5% False)
    - Zero data leakage: Scaler & OneHotEncoder are fitted ONLY on X_train.
    - Scaled representations available with feature names aligned.
    """
    raw_df = load_raw_data()
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
