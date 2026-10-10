"""
test_app.py — Automated verification test suite for Streamlit ML Capstone App.
Uses streamlit.testing.v1.AppTest to exercise all pages, forms, and predictions.
"""

import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent
APP_FILE = str(ROOT / "app.py")


def test_home_page():
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert not at.exception
    assert at.session_state.page == "Home"
    assert len(at.button) >= 3


def test_navigation_to_regression():
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    btn_reg = [b for b in at.button if "Explore Regression" in b.label][0]
    btn_reg.click().run()
    assert not at.exception
    assert at.session_state.page == "Regression"


def test_regression_prediction():
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    at.radio(key="nav_radio").set_value("Regression").run()
    form_btn = [b for b in at.button if "Predict Daily Sales Volume" in b.label][0]
    form_btn.click().run()
    assert not at.exception
    pred_md = [m.value for m in at.markdown if "pred-value" in m.value and "&#xa3;" in m.value]
    assert len(pred_md) > 0


def test_classification_prediction():
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    at.radio(key="nav_radio").set_value("Classification").run()
    clf_btn = [b for b in at.button if "Predict Purchase Intent" in b.label][0]
    clf_btn.click().run()
    assert not at.exception
    pred_md = [m.value for m in at.markdown if "pred-value" in m.value]
    assert len(pred_md) > 0


def test_clustering_page_missing_artifacts_warning():
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    at.radio(key="nav_radio").set_value("Clustering").run()
    assert not at.exception
    warning_boxes = [m.value for m in at.markdown if "error-box" in m.value or "Model artifacts not found" in m.value]
    assert len(warning_boxes) > 0


def test_model_comparison_page():
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    at.radio(key="nav_radio").set_value("Model Comparison").run()
    assert not at.exception
    assert len(at.dataframe) >= 3


def test_about_page():
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    at.radio(key="nav_radio").set_value("About").run()
    assert not at.exception
    assert len(at.code) >= 2


if __name__ == "__main__":
    test_home_page()
    test_navigation_to_regression()
    test_regression_prediction()
    test_classification_prediction()
    test_clustering_page_missing_artifacts_warning()
    test_model_comparison_page()
    test_about_page()
    print("All test_app.py tests completed successfully!")
