"""
export_models.py — Train and save all ML model artifacts to models/

Run from the project root:
    python export_models.py

Creates:
    models/regression_preprocessor.pkl
    models/regression_model.pkl        (Tuned Random Forest Regressor)
    models/regression_outlier_bounds.pkl
    models/classification_preprocessor.pkl  (scaler + encoder from nb_utils)
    models/classification_model.pkl    (SVM RBF, C=10)
    models/clustering_scaler.pkl
    models/clustering_model.pkl        (MiniBatchKMeans, K=4)
    models/cluster_profile.pkl
    models/cluster_summary.pkl
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "notebooks"))

# ─────────────────────────── REGRESSION ────────────────────────────────────

def export_regression():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor

    print("Exporting regression model …")

    df = pd.read_csv(ROOT / "data" / "raw" / "data_1.csv", encoding="ISO-8859-1")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek
    df.drop(columns=["Date"], inplace=True)

    df["avg_items_per_order"] = df["total_items_sold"] / df["total_orders"]

    TARGET = "daily_sales_volume"
    X = df.drop(columns=[TARGET])
    y_original = df[TARGET].clip(lower=0)
    y = np.log1p(y_original)

    continuous_features = [
        "product_variation_count", "total_items_sold",
        "total_orders", "avg_items_per_order",
    ]

    y_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = train_test_split(
        X, y, y_original, test_size=0.20, random_state=42, stratify=y_bins
    )

    # Clip outliers using training IQR bounds
    outlier_bounds = {}
    for col in continuous_features:
        if col not in X_train.columns:
            continue
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lb = Q1 - 1.5 * IQR
        ub = Q3 + 1.5 * IQR
        outlier_bounds[col] = (lb, ub)
        X_train[col] = X_train[col].clip(lb, ub)
        X_test[col] = X_test[col].clip(lb, ub)

    # Cap training target outliers
    Q1_y = y_train.quantile(0.25)
    Q3_y = y_train.quantile(0.75)
    IQR_y = Q3_y - Q1_y
    y_train = y_train.clip(Q1_y - 1.5 * IQR_y, Q3_y + 1.5 * IQR_y)

    # Group rare countries
    country_counts = X_train["Country"].value_counts()
    rare = country_counts[country_counts < 5].index
    X_train["Country"] = X_train["Country"].replace(rare, "Other")
    X_test["Country"] = X_test["Country"].replace(rare, "Other")

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)

    model = RandomForestRegressor(
        n_estimators=100, max_depth=20, max_features="sqrt",
        min_samples_leaf=1, min_samples_split=2, random_state=42, n_jobs=-1
    )
    model.fit(X_train_proc, y_train)

    joblib.dump(preprocessor, MODELS_DIR / "regression_preprocessor.pkl")
    joblib.dump(model, MODELS_DIR / "regression_model.pkl")
    joblib.dump(outlier_bounds, MODELS_DIR / "regression_outlier_bounds.pkl")
    print("  [OK] Regression artifacts saved.")


# ─────────────────────────── CLASSIFICATION ────────────────────────────────

def export_classification():
    from nb_utils import load_and_preprocess_data
    from sklearn.svm import SVC

    print("Exporting classification model …")

    data = load_and_preprocess_data(test_size=0.20, random_state=42)
    X_train = data["X_train"]
    y_train = data["y_train"]
    scaler = data["scaler"]
    encoder = data["encoder"]

    model = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
    model.fit(X_train, y_train)

    joblib.dump({"scaler": scaler, "encoder": encoder}, MODELS_DIR / "classification_preprocessor.pkl")
    joblib.dump(model, MODELS_DIR / "classification_model.pkl")
    print("  [OK] Classification artifacts saved.")


# ─────────────────────────── CLUSTERING ────────────────────────────────────

def export_clustering():
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import MiniBatchKMeans

    print("Exporting clustering model (full pandas load with memory optimization) …")

    df = pd.read_csv(ROOT / "data" / "raw" / "2019-Oct.csv", low_memory=False)

    df_clean = df.drop_duplicates()
    df_clean = df_clean.dropna(subset=["user_session"]).copy()
    df_clean["event_time"] = pd.to_datetime(df_clean["event_time"], errors="coerce")
    df_clean["is_view"] = (df_clean["event_type"] == "view").astype("int8")
    df_clean["is_cart"] = (df_clean["event_type"] == "cart").astype("int8")
    df_clean["cart_product_id"] = df_clean["product_id"].where(df_clean["event_type"] == "cart")
    df_clean["view_price"] = df_clean["price"].where(df_clean["event_type"] == "view")

    print("  Aggregating session features …")
    session_features = (
        df_clean
        .groupby("user_session", sort=False)
        .agg(
            unique_products_viewed=("product_id", "nunique"),
            unique_categories_viewed=("category_id", "nunique"),
            total_views=("is_view", "sum"),
            cart_additions=("is_cart", "sum"),
            unique_products_carted=("cart_product_id", "nunique"),
            avg_price_viewed=("view_price", "mean"),
            max_price_viewed=("view_price", "max"),
            session_start=("event_time", "min"),
            session_end=("event_time", "max"),
        )
        .reset_index()
    )

    del df, df_clean

    session_features["session_duration_minutes"] = (
        (session_features["session_end"] - session_features["session_start"])
        .dt.total_seconds() / 60
    )

    cluster_features = [
        "unique_products_viewed", "unique_categories_viewed", "total_views",
        "cart_additions", "unique_products_carted", "avg_price_viewed",
        "max_price_viewed", "session_duration_minutes",
    ]

    X_cluster = session_features[cluster_features].replace([np.inf, -np.inf], np.nan).fillna(0)

    print("  Fitting scaler and training MiniBatchKMeans …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    K = 4
    RANDOM_STATE = 42
    mini_kmeans = MiniBatchKMeans(n_clusters=K, batch_size=2048, random_state=RANDOM_STATE, n_init=10)
    mini_kmeans.fit(X_scaled)

    session_features["cluster"] = mini_kmeans.predict(X_scaled)

    cluster_profile = (
        session_features
        .groupby("cluster")[cluster_features]
        .mean()
        .round(2)
    )

    cluster_sizes = session_features["cluster"].value_counts().sort_index()
    cluster_pct = (
        session_features["cluster"].value_counts(normalize=True).sort_index().mul(100)
    )
    cluster_summary = pd.DataFrame({"Sessions": cluster_sizes, "Percentage (%)": cluster_pct.round(2)})

    joblib.dump(scaler, MODELS_DIR / "clustering_scaler.pkl")
    joblib.dump(mini_kmeans, MODELS_DIR / "clustering_model.pkl")
    joblib.dump(cluster_profile, MODELS_DIR / "cluster_profile.pkl")
    joblib.dump(cluster_summary, MODELS_DIR / "cluster_summary.pkl")

    print("  [OK] Clustering artifacts saved.")
    print(f"  Notebook cluster sizes for comparison: 0=1,431,933, 1=6,770,409, 2=486,228, 3=555,851")
    print(f"  Generated cluster sizes: {cluster_sizes.to_dict()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["regression", "classification", "clustering", "all"], default="all")
    args = parser.parse_args()

    if args.task in ("regression", "all"):
        export_regression()
    if args.task in ("classification", "all"):
        export_classification()
    if args.task in ("clustering", "all"):
        export_clustering()

    print("\nAll requested artifacts saved to:", MODELS_DIR)
