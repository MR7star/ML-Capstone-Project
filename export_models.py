"""
export_models.py — Train and save all ML model artifacts to models/

Run from the project root:
    python export_models.py

Creates:
    models/regression_preprocessor.pkl
    models/regression_model.pkl        (Tuned Random Forest Regressor)
    models/regression_outlier_bounds.pkl
    models/classification_preprocessor.pkl  (scaler + encoder from nb_utils)
    models/classification_model_logistic.pkl
    models/classification_model_knn.pkl
    models/classification_model_nb.pkl
    models/classification_model_dt.pkl
    models/classification_model_svm.pkl
    models/classification_model_rf.pkl
    models/classification_model_ada.pkl
    models/classification_model_gb.pkl
    models/classification_model_bag.pkl
    models/classification_model_mlp.pkl
    models/clustering_scaler.pkl
    models/clustering_model.pkl        (MiniBatchKMeans, K=4)
    models/cluster_profile.pkl
    models/cluster_summary.pkl
"""

import sys
import os
import shutil
import tempfile
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
    from sklearn.neural_network import MLPClassifier

    print("Exporting classification models …")

    data = load_and_preprocess_data(test_size=0.20, random_state=42)
    X_train = data["X_train"]
    y_train = data["y_train"]
    scaler = data["scaler"]
    encoder = data["encoder"]

    joblib.dump({"scaler": scaler, "encoder": encoder}, MODELS_DIR / "classification_preprocessor.pkl")

    models = {
        "logistic": LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'),
        "knn": KNeighborsClassifier(n_neighbors=13, metric="manhattan"),
        "nb": GaussianNB(var_smoothing=1e-3),
        "dt": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
        "svm": SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42, probability=True),
        "rf": RandomForestClassifier(n_estimators=200, max_depth=20, max_features="sqrt", min_samples_leaf=1, random_state=42, n_jobs=-1),
        "ada": AdaBoostClassifier(n_estimators=50, learning_rate=0.5, random_state=42),
        "gb": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
        "bag": BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=100, max_samples=1.0, max_features=1.0, random_state=42, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(100,), activation="relu", alpha=0.001, max_iter=500, early_stopping=True, random_state=42),
    }

    for name, model in models.items():
        print(f"  Training {name} …")
        model.fit(X_train, y_train)
        joblib.dump(model, MODELS_DIR / f"classification_model_{name}.pkl")
        print(f"  [OK] {name} saved.")

    print("  [OK] All classification artifacts saved.")


# ─────────────────────────── CLUSTERING ────────────────────────────────────

def export_clustering():
    import duckdb
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import MiniBatchKMeans

    print("Exporting clustering model using DuckDB global deduplication …")
    cluster_features = [
        "unique_products_viewed", "unique_categories_viewed", "total_views",
        "cart_additions", "unique_products_carted", "avg_price_viewed",
        "max_price_viewed", "session_duration_minutes",
    ]

    raw_path = ROOT / "data" / "raw" / "2019-Oct.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Clustering source dataset not found: {raw_path}")

    # Keep DuckDB spill files and candidate artifacts separate from the live artifacts.
    with tempfile.TemporaryDirectory(prefix="clustering-export-") as work_dir:
        spill_dir = Path(work_dir) / "duckdb_spill"
        spill_dir.mkdir()
        candidate_dir = Path(tempfile.mkdtemp(prefix=".clustering-candidate-", dir=MODELS_DIR))
        con = None
        try:
            con = duckdb.connect(config={
                "memory_limit": "4GB",
                "temp_directory": str(spill_dir),
                "preserve_insertion_order": "true",
                "threads": "4",
            })
            columns = [
                "event_time", "event_type", "product_id", "category_id",
                "category_code", "brand", "price", "user_id", "user_session",
            ]
            # Grouping by every input field gives drop_duplicates() global semantics;
            # min(ordinality) retains the first input row for notebook-compatible order.
            con.execute(f"""
                CREATE TEMP TABLE deduplicated_events AS
                SELECT {", ".join(columns)},
                       MIN(ordinality) AS source_order,
                       COUNT(*) AS source_copies
                FROM read_csv_auto(
                    ?, sample_size=100000,
                    nullstr=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                             '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                             '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None',
                             'n/a', 'nan', 'null']
                ) WITH ORDINALITY
                GROUP BY {", ".join(columns)}
            """, [str(raw_path)])

            raw_rows, deduplicated_rows, missing_sessions, clean_rows = con.execute("""
                SELECT SUM(source_copies), COUNT(*),
                       COUNT(*) FILTER (WHERE user_session IS NULL),
                       COUNT(*) FILTER (WHERE user_session IS NOT NULL)
                FROM deduplicated_events
            """).fetchone()
            duplicate_rows = raw_rows - deduplicated_rows
            print(
                "  Event rows: "
                f"raw={raw_rows:,}, after global drop_duplicates={deduplicated_rows:,}, "
                f"duplicates removed={duplicate_rows:,}, missing sessions removed="
                f"{missing_sessions:,}, cleaned={clean_rows:,}"
            )

            print("  Aggregating session features …")
            session_features = con.execute("""
                SELECT
                    user_session,
                    COUNT(DISTINCT product_id) AS unique_products_viewed,
                    COUNT(DISTINCT category_id) AS unique_categories_viewed,
                    SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS total_views,
                    SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS cart_additions,
                    COUNT(DISTINCT CASE WHEN event_type = 'cart' THEN product_id END)
                        AS unique_products_carted,
                    AVG(CASE WHEN event_type = 'view' THEN price END) AS avg_price_viewed,
                    MAX(CASE WHEN event_type = 'view' THEN price END) AS max_price_viewed,
                    DATE_DIFF(
                        'microseconds',
                        MIN(TRY_CAST(event_time AS TIMESTAMP)),
                        MAX(TRY_CAST(event_time AS TIMESTAMP))
                    ) / 60000000.0 AS session_duration_minutes
                FROM deduplicated_events
                WHERE user_session IS NOT NULL
                GROUP BY user_session
                ORDER BY MIN(source_order)
            """).fetchdf()
            con.close()

            session_count = len(session_features)
            print(f"  Aggregated sessions: {session_count:,} (notebook: 9,244,421)")
            if session_count == 0:
                raise ValueError("No sessions remain after clustering preprocessing.")

            X_cluster = (
                session_features[cluster_features]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )

            print("  Fitting scaler and training MiniBatchKMeans …")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)

            K = 4
            RANDOM_STATE = 42
            mini_kmeans = MiniBatchKMeans(
                n_clusters=K, batch_size=2048, random_state=RANDOM_STATE, n_init=10
            )
            mini_kmeans.fit(X_scaled)

            session_features["cluster"] = mini_kmeans.labels_
            cluster_profile = (
                session_features.groupby("cluster")[cluster_features].mean().round(2)
            )
            cluster_sizes = session_features["cluster"].value_counts().sort_index()
            cluster_pct = (
                session_features["cluster"].value_counts(normalize=True)
                .sort_index().mul(100)
            )
            cluster_summary = pd.DataFrame({
                "Sessions": cluster_sizes,
                "Percentage (%)": cluster_pct.round(2),
            })

            if (
                len(cluster_features) != mini_kmeans.n_features_in_
                or cluster_sizes.sum() != session_count
                or not np.isfinite(X_scaled).all()
            ):
                raise ValueError("Clustering candidate failed feature or session-count validation.")

            artifacts = {
                "clustering_scaler.pkl": scaler,
                "clustering_model.pkl": mini_kmeans,
                "cluster_profile.pkl": cluster_profile,
                "cluster_summary.pkl": cluster_summary,
            }
            for name, artifact in artifacts.items():
                joblib.dump(artifact, candidate_dir / name)

            validated = {
                name: joblib.load(candidate_dir / name)
                for name in artifacts
            }
            probe = validated["clustering_scaler.pkl"].transform(X_cluster.iloc[:5])
            predictions = validated["clustering_model.pkl"].predict(probe)
            if (
                validated["clustering_model.pkl"].n_clusters != K
                or validated["clustering_model.pkl"].n_features_in_ != len(cluster_features)
                or len(predictions) != min(5, session_count)
                or not set(predictions).issubset(set(range(K)))
                or int(validated["cluster_summary.pkl"]["Sessions"].sum()) != session_count
                or not validated["cluster_profile.pkl"].index.equals(cluster_profile.index)
            ):
                raise ValueError("Clustering candidate artifacts failed reload/prediction validation.")

            documented_sizes = {
                0: 1431933, 1: 6770409, 2: 486228, 3: 555851,
            }
            print(f"  Notebook cluster sizes for comparison: {documented_sizes}")
            print(f"  Generated cluster sizes (unmodified model labels): {cluster_sizes.to_dict()}")
            for name in artifacts:
                os.replace(candidate_dir / name, MODELS_DIR / name)
            print("  [OK] Validated clustering artifacts saved.")
        finally:
            if con is not None:
                con.close()
            shutil.rmtree(candidate_dir)


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
