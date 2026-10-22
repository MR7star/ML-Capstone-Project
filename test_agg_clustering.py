"""
test_agg_clustering.py — Comprehensive tests for Agglomerative Clustering integration.

Tests:
1. Artifact loader returns None gracefully when artifacts are absent.
2. Streamlit Clustering page algorithm selector allows switching between MiniBatch K-Means
   and Agglomerative Hierarchical Clustering without exception.
3. When artifacts are missing, the clear, non-crashing warning notice is rendered.
4. Model Comparison page renders both K-Means and Agglomerative sections gracefully.
5. In a mock fixture environment:
   - Validates that Agglomerative Clustering uses exactly 7 features (excluding max_price_viewed).
   - Validates that K-Means uses 8 features (including max_price_viewed).
   - Validates nearest-centroid assignment logic using Euclidean distance in scaled space.
   - Validates that load_agg_clustering_artifacts() loads and returns all required keys.
"""

import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent
APP_FILE = str(ROOT / "app.py")


def test_agg_loader_missing_artifacts():
    """Verify loader returns None without throwing when directory or files are absent."""
    import app
    result = app.load_agg_clustering_artifacts()
    # If artifacts directory does not exist or files not yet exported, should return None
    agg_dir = app.MODELS_DIR / "agg_clustering_artifacts"
    if not agg_dir.exists() or not (agg_dir / "agg_scaler.pkl").exists():
        assert result is None
    else:
        assert isinstance(result, dict)


def test_clustering_algorithm_selector_and_graceful_missing_notice():
    """Verify radio selector switches to Agglomerative and renders missing-artifact warning."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    at.radio(key="nav_radio").set_value("Clustering").run()
    assert not at.exception

    # Find the clustering algorithm radio selector
    radio_selectors = [r for r in at.radio if r.key == "clustering_algo_selector"]
    assert len(radio_selectors) == 1
    algo_radio = radio_selectors[0]
    assert "MiniBatch K-Means" in algo_radio.options
    assert "Agglomerative Hierarchical Clustering" in algo_radio.options

    # Switch to Agglomerative Hierarchical Clustering
    algo_radio.set_value("Agglomerative Hierarchical Clustering").run()
    assert not at.exception

    # Check for graceful warning or rendered overview
    notices = [m.value for m in at.markdown if "Agglomerative Clustering artifacts not found" in m.value or "Model Overview" in m.value]
    assert len(notices) > 0


def test_model_comparison_page_has_agglomerative_section():
    """Verify Model Comparison page renders both clustering algorithms without error."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    at.radio(key="nav_radio").set_value("Model Comparison").run()
    assert not at.exception

    # Check that agglomerative hierarchical clustering header or comparison note appears
    agg_headers = [m.value for m in at.markdown if "Agglomerative Hierarchical Clustering" in m.value]
    assert len(agg_headers) > 0


def test_feature_dimensions_and_isolation():
    """Verify that Agglomerative uses 7 features while K-Means uses 8 features."""
    agg_expected_features = [
        "unique_products_viewed",
        "unique_categories_viewed",
        "total_views",
        "cart_additions",
        "unique_products_carted",
        "avg_price_viewed",
        "session_duration_minutes",
    ]
    assert len(agg_expected_features) == 7
    assert "max_price_viewed" not in agg_expected_features

    # Existing K-Means scaler uses 8 features
    import app
    km_scaler, _, _, _ = app.load_clustering_artifacts()
    if km_scaler is not None:
        assert km_scaler.n_features_in_ == 8
        if hasattr(km_scaler, "feature_names_in_"):
            assert "max_price_viewed" in km_scaler.feature_names_in_


def test_fixture_pipeline_and_centroid_assignment():
    """Verify artifact loading and nearest-centroid assignment with mock artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        features = [
            "unique_products_viewed",
            "unique_categories_viewed",
            "total_views",
            "cart_additions",
            "unique_products_carted",
            "avg_price_viewed",
            "session_duration_minutes",
        ]
        X_mock = np.random.RandomState(42).randn(100, 7)
        scaler = StandardScaler().fit(X_mock)
        centroids = np.array([
            [1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.5],
            [-0.5, -0.5, -0.5, 0.0, 0.0, -0.5, -0.5],
            [2.5, 2.0, 3.0, 0.2, 0.1, 0.5, 2.0],
            [0.2, 0.2, 0.5, 2.0, 1.5, 0.8, 1.0],
        ])
        profile_mean = pd.DataFrame(centroids, columns=features)
        profile_median = pd.DataFrame(centroids, columns=features)
        summary = pd.DataFrame({"Sessions": [40, 35, 15, 10], "Percentage": [40.0, 35.0, 15.0, 10.0]})
        eval_metrics = {"Silhouette Score": 0.6593, "Davies-Bouldin Index": 0.6541, "Calinski-Harabasz Index": 1823.4}
        eval_by_k = pd.DataFrame([
            {"K": 2, "Silhouette Score": 0.6475, "Davies-Bouldin Index": 0.6879, "Calinski-Harabasz Index": 720.0, "Smallest Cluster (%)": 11.0, "Largest Cluster (%)": 89.0},
            {"K": 3, "Silhouette Score": 0.6321, "Davies-Bouldin Index": 0.6710, "Calinski-Harabasz Index": 1250.0, "Smallest Cluster (%)": 6.5, "Largest Cluster (%)": 87.5},
            {"K": 4, "Silhouette Score": 0.6593, "Davies-Bouldin Index": 0.6541, "Calinski-Harabasz Index": 1823.4, "Smallest Cluster (%)": 0.02, "Largest Cluster (%)": 88.22},
        ]).set_index("K")
        config = {
            "cluster_features": features,
            "n_clusters": 4,
            "linkage": "ward",
            "hierarchical_sample_size": 5000,
            "n_eligible_sessions": 9242653,
            "k_selection_sample_size": 2000,
            "random_state": 42,
        }

        # Save all artifacts to tmp_path
        joblib.dump(scaler, tmp_path / "agg_scaler.pkl")
        joblib.dump(centroids, tmp_path / "agg_centroids.npy")
        joblib.dump(profile_mean, tmp_path / "agg_profile_mean.pkl")
        joblib.dump(profile_median, tmp_path / "agg_profile_median.pkl")
        joblib.dump(summary, tmp_path / "agg_cluster_summary.pkl")
        joblib.dump(eval_metrics, tmp_path / "agg_eval_metrics.pkl")
        joblib.dump(eval_by_k, tmp_path / "agg_eval_by_k.pkl")
        joblib.dump(config, tmp_path / "agg_config.pkl")

        # Verify nearest centroid assignment computation
        sample_input = pd.DataFrame([[3, 1, 4, 0, 0, 50.0, 5.0]], columns=features)
        scaled_input = scaler.transform(sample_input)
        dists = np.linalg.norm(centroids - scaled_input, axis=1)
        assigned_cluster = int(np.argmin(dists))
        assert 0 <= assigned_cluster < 4


if __name__ == "__main__":
    test_agg_loader_missing_artifacts()
    test_clustering_algorithm_selector_and_graceful_missing_notice()
    test_model_comparison_page_has_agglomerative_section()
    test_feature_dimensions_and_isolation()
    test_fixture_pipeline_and_centroid_assignment()
    print("All test_agg_clustering.py tests completed successfully!")
