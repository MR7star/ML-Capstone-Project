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


def title(text: str, kicker: str = None):
    """
    Renders a clean, modern cell header with an optional kicker tag.
    Directly addresses user requirement: title("...", kicker="...")
    """
    kicker_html = f"<div style='font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: {ACCENT}; margin-bottom: 4px;'>{kicker}</div>" if kicker else ""
    html = f"""
    <div style='margin-top: 14px; margin-bottom: 12px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
        {kicker_html}
        <h2 style='margin: 0; font-size: 20px; font-weight: 700; color: {DARK}; letter-spacing: -0.02em;'>{text}</h2>
        <hr style='border: none; border-top: 1.5px solid #e2e8f0; margin-top: 8px; margin-bottom: 0;'/>
    </div>
    """
    try:
        display(HTML(html))
    except Exception:
        prefix = f"[{kicker.upper()}] " if kicker else ""
        print(f"\n{'='*60}\n{prefix}{text}\n{'='*60}")


def hero(cards: list):
    """
    Renders high-impact metric KPI cards.
    cards: list of tuples -> (stat_value, label, color)
    Example: hero([("89.4%", "Test Accuracy", "#16a34a"), ("0.654", "Weighted F1", "#2563eb")])
    """
    card_htmls = []
    for stat, label, color in cards:
        card = f"""
        <div style='flex: 1; min-width: 150px; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-right: 12px; margin-bottom: 8px;'>
            <div style='font-size: 22px; font-weight: 800; color: {color}; line-height: 1.2;'>{stat}</div>
            <div style='font-size: 12px; font-weight: 600; color: {MUTED}; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.04em;'>{label}</div>
        </div>
        """
        card_htmls.append(card)

    container = f"""
    <div style='display: flex; flex-wrap: wrap; margin: 12px 0 16px 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
        {''.join(card_htmls)}
    </div>
    """
    try:
        display(HTML(container))
    except Exception:
        formatted = " | ".join([f"{label}: {stat}" for stat, label, _ in cards])
        print(f"\n>>> [METRICS] {formatted}\n")


def finish(fig, title_text: str, note: str = None):
    """
    Standardizes plot aesthetics, titles, tight_layout, and optional explanatory notes.
    """
    fig.suptitle(title_text, fontsize=13, fontweight="bold", y=1.02, color=DARK)
    if note:
        fig.text(0.5, -0.04, note, ha="center", fontsize=9.5, color=MUTED, style="italic")
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
