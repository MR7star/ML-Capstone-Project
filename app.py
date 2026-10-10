import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

st.set_page_config(
    page_title="ML Capstone Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"

# ─────────────────────────── PASTEL DESIGN SYSTEM CSS ────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #303247;
}

.stApp {
    background-color: #F7F6FC;
}

[data-testid="stSidebar"] {
    background-color: #E5EAF8;
    border-right: 1px solid #E6E4EF;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 0.45rem 0.75rem;
    border-radius: 8px;
    transition: background-color 0.15s ease, color 0.15s ease;
    margin-bottom: 2px;
    cursor: pointer;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: rgba(139, 131, 201, 0.12);
}

h1, h2, h3, h4, h5, h6 {
    color: #303247;
    font-weight: 600;
    letter-spacing: -0.015em;
}

p, span, label {
    color: #303247;
}

.custom-card {
    background-color: #FFFFFF;
    border: 1px solid #E6E4EF;
    border-radius: 12px;
    padding: 1.35rem 1.5rem;
    box-shadow: 0 1px 3px rgba(48, 50, 71, 0.03);
    margin-bottom: 1rem;
}

.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #E6E4EF;
    border-radius: 10px;
    padding: 1.1rem 0.85rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(48, 50, 71, 0.03);
    min-height: 96px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #8B83C9;
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}

.metric-card .label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #73758A;
    margin-top: 5px;
    line-height: 1.3;
}

.hero-card {
    background-color: #FFFFFF;
    border: 1px solid #E6E4EF;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 4px rgba(48, 50, 71, 0.03);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 255px;
    margin-bottom: 0.75rem;
}

.hero-card h3 {
    font-size: 1.15rem;
    margin: 0.45rem 0 0.35rem;
    color: #303247;
}

.hero-card p {
    color: #73758A;
    font-size: 0.88rem;
    line-height: 1.55;
    margin-bottom: 0.4rem;
}

.cluster-card {
    background-color: #FFFFFF;
    border: 1px solid #E6E4EF;
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 1px 3px rgba(48, 50, 71, 0.03);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 180px;
    margin-bottom: 0.75rem;
}

.cluster-card h4 {
    margin: 0.45rem 0 0.3rem;
    font-size: 1rem;
    color: #303247;
}

.cluster-card p {
    font-size: 0.83rem;
    color: #73758A;
    line-height: 1.5;
    margin: 0;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    width: fit-content;
}

.badge-regression {
    background-color: rgba(169, 185, 232, 0.3);
    color: #40507a;
}

.badge-classification {
    background-color: rgba(168, 201, 181, 0.35);
    color: #2b553e;
}

.badge-clustering {
    background-color: rgba(232, 185, 168, 0.45);
    color: #7b4231;
}

.badge-accent {
    background-color: rgba(139, 131, 201, 0.25);
    color: #554d8a;
}

.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #303247;
    border-left: 3px solid #8B83C9;
    padding-left: 0.65rem;
    margin: 1.35rem 0 0.85rem;
    line-height: 1.35;
}

.result-card {
    background-color: #FFFFFF;
    border: 1.5px solid #8B83C9;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(139, 131, 201, 0.08);
    margin: 1.25rem 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.result-card.success {
    background-color: rgba(168, 201, 181, 0.18);
    border-color: #A8C9B5;
}

.result-card.warning {
    background-color: rgba(232, 185, 168, 0.22);
    border-color: #E8B9A8;
}

.result-card .res-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #73758A;
}

.result-card .res-value {
    font-size: 2.15rem;
    font-weight: 700;
    color: #303247;
    margin: 0.35rem 0;
}

.result-card.success .res-value {
    color: #28573d;
}

.result-card.warning .res-value {
    color: #8c4633;
}

.result-card .res-desc {
    font-size: 0.88rem;
    color: #73758A;
    margin-top: 0.35rem;
}

.pastel-notice {
    background-color: rgba(232, 185, 168, 0.22);
    border: 1px solid #E8B9A8;
    border-radius: 10px;
    padding: 1.1rem 1.35rem;
    color: #73758A;
    margin: 1rem 0;
    line-height: 1.55;
}

.pastel-notice b {
    color: #303247;
}

.pastel-notice code {
    background-color: #FFFFFF;
    color: #303247;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    border: 1px solid #E6E4EF;
    font-size: 0.85rem;
}

.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    border: 1px solid #E6E4EF;
    background-color: #FFFFFF;
    color: #303247;
    transition: all 0.15s ease;
    padding: 0.5rem 1rem;
}

.stButton > button:hover {
    border-color: #8B83C9;
    color: #8B83C9;
    background-color: #F7F6FC;
}

button[kind="primary"], div.stFormSubmitButton > button {
    background-color: #8B83C9 !important;
    color: #FFFFFF !important;
    border: 1px solid #8B83C9 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.55rem 1rem !important;
}

button[kind="primary"]:hover, div.stFormSubmitButton > button:hover {
    background-color: #7b73b8 !important;
    border-color: #7b73b8 !important;
    color: #FFFFFF !important;
}

[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #E6E4EF;
    border-radius: 12px;
    padding: 1.5rem;
}

hr {
    border-color: #E6E4EF !important;
    margin: 1.25rem 0 !important;
}
</style>
""", unsafe_allow_html=True)


def configure_matplotlib_theme():
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#E6E4EF',
        'axes.linewidth': 1.0,
        'axes.labelcolor': '#303247',
        'axes.labelsize': 9.5,
        'axes.titlesize': 10.5,
        'axes.titleweight': 'bold',
        'axes.titlecolor': '#303247',
        'axes.titlepad': 10,
        'xtick.color': '#73758A',
        'ytick.color': '#73758A',
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'grid.color': '#F2F1F8',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.8,
        'text.color': '#303247',
        'font.sans-serif': ['Inter', 'Segoe UI', 'DejaVu Sans', 'sans-serif'],
    })


def render_page_header(title: str, subtitle: str):
    st.markdown(
        f'<div style="margin-bottom: 0.5rem;">'
        f'<h1 style="margin: 0 0 0.35rem 0; font-size: 1.85rem; font-weight: 700; color: #303247; letter-spacing: -0.02em;">{title}</h1>'
        f'<p style="margin: 0; color: #73758A; font-size: 0.95rem; line-height: 1.5;">{subtitle}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()


PAGES = ["Home", "Regression", "Classification", "Clustering", "Model Comparison", "About"]

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "nav_radio" not in st.session_state:
    st.session_state.nav_radio = "Home"


def go_to(target):
    st.session_state.page = target
    st.session_state.nav_radio = target


def _on_radio_change():
    st.session_state.page = st.session_state.nav_radio


with st.sidebar:
    st.markdown("### ML Capstone")
    st.caption("E-Commerce Predictive & Clustering Suite")
    st.markdown("<hr style='margin:0.5rem 0 1rem'>", unsafe_allow_html=True)
    st.radio("Navigate to", PAGES, key="nav_radio", on_change=_on_radio_change)
    st.markdown("<hr style='margin:1.5rem 0 1rem'>", unsafe_allow_html=True)
    st.caption("Python 3.12 | scikit-learn 1.8 | Streamlit")

page = st.session_state.page


@st.cache_resource(show_spinner=False)
def load_regression_artifacts():
    try:
        pre = joblib.load(MODELS_DIR / "regression_preprocessor.pkl")
        mdl = joblib.load(MODELS_DIR / "regression_model.pkl")
        bounds = joblib.load(MODELS_DIR / "regression_outlier_bounds.pkl")
        return pre, mdl, bounds
    except FileNotFoundError:
        return None, None, None


@st.cache_resource(show_spinner=False)
def load_classification_artifacts():
    try:
        pre = joblib.load(MODELS_DIR / "classification_preprocessor.pkl")
        mdl = joblib.load(MODELS_DIR / "classification_model.pkl")
        return pre, mdl
    except FileNotFoundError:
        return None, None


@st.cache_resource(show_spinner=False)
def load_clustering_artifacts():
    try:
        scaler = joblib.load(MODELS_DIR / "clustering_scaler.pkl")
        mdl = joblib.load(MODELS_DIR / "clustering_model.pkl")
        profile = joblib.load(MODELS_DIR / "cluster_profile.pkl")
        summary = joblib.load(MODELS_DIR / "cluster_summary.pkl")
        return scaler, mdl, profile, summary
    except FileNotFoundError:
        return None, None, None, None


def artifact_missing_warning(task_name):
    st.markdown(
        f'<div class="pastel-notice error-box"><b>Model artifacts not found for {task_name}.</b><br>'
        f'To generate artifacts from the raw dataset, run: <code>python export_models.py --task {task_name.lower()}</code> '
        f'from the project root directory.</div>',
        unsafe_allow_html=True,
    )


# ════════════════════ HOME ═══════════════════════════════════════════════════

if page == "Home":
    render_page_header(
        "Machine Learning Capstone",
        "An interactive machine learning dashboard analyzing e-commerce transactions across "
        "<b>Regression</b>, <b>Classification</b>, and <b>Clustering</b> domains."
    )

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown("""<div class="hero-card">
            <div>
                <span class="badge badge-regression">TRACK 1 &bull; REGRESSION</span>
                <h3>Daily Sales Volume</h3>
                <p>Predicts daily total sales volume in GBP (&pound;) based on catalog SKU variation, daily order count, total items sold, and calendar context.</p>
            </div>
            <p style="margin-top:0.75rem;padding-top:0.5rem;border-top:1px solid #E6E4EF;">
                <b>Model:</b> Tuned Random Forest<br>
                <span style="color:#73758A;">R&sup2; Score: <b>0.892</b> &bull; RMSE: <b>&pound;3,940</b></span>
            </p>
        </div>""", unsafe_allow_html=True)
        st.button("Explore Regression →", key="btn_reg", use_container_width=True,
                  on_click=go_to, args=("Regression",))

    with col2:
        st.markdown("""<div class="hero-card">
            <div>
                <span class="badge badge-classification">TRACK 2 &bull; CLASSIFICATION</span>
                <h3>Purchase Intent</h3>
                <p>Determines whether an online shopping session converts into a completed purchase (Revenue = True) from multi-page behavioral features.</p>
            </div>
            <p style="margin-top:0.75rem;padding-top:0.5rem;border-top:1px solid #E6E4EF;">
                <b>Model:</b> SVM (RBF, C=10)<br>
                <span style="color:#73758A;">Weighted F1: <b>0.883</b> &bull; ROC-AUC: <b>0.863</b></span>
            </p>
        </div>""", unsafe_allow_html=True)
        st.button("Explore Classification →", key="btn_clf", use_container_width=True,
                  on_click=go_to, args=("Classification",))

    with col3:
        st.markdown("""<div class="hero-card">
            <div>
                <span class="badge badge-clustering">TRACK 3 &bull; CLUSTERING</span>
                <h3>Customer Segmentation</h3>
                <p>Segments 9.24 million e-commerce user sessions into 4 distinct behavioural groups based on browsing depth, carting, and price views.</p>
            </div>
            <p style="margin-top:0.75rem;padding-top:0.5rem;border-top:1px solid #E6E4EF;">
                <b>Model:</b> MiniBatch K-Means (K=4)<br>
                <span style="color:#73758A;">Silhouette: <b>0.542</b> &bull; Calinski-H: <b>15,991</b></span>
            </p>
        </div>""", unsafe_allow_html=True)
        st.button("Explore Clustering →", key="btn_clust", use_container_width=True,
                  on_click=go_to, args=("Clustering",))

    st.markdown('<p class="section-header">Verified Model Performance Highlights</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    for col, val, lbl in [
        (c1, "0.892", "Regression R²"),
        (c2, "£3,940", "Regression RMSE"),
        (c3, "89.0%", "Classifier Accuracy"),
        (c4, "0.883", "Classifier F1"),
        (c5, "0.863", "Classifier AUC"),
        (c6, "0.542", "Cluster Silhouette (K=4)"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="value">{val}</div><div class="label">{lbl}</div></div>',
            unsafe_allow_html=True,
        )


# ════════════════════ REGRESSION ═════════════════════════════════════════════

elif page == "Regression":
    render_page_header(
        "Daily Sales Volume Prediction",
        "Predict expected daily sales revenue (&pound;) for country and date combinations. "
        "Built with a <b>Tuned Random Forest Regressor</b> using IQR outlier clipping and log1p target scaling."
    )

    preprocessor, model, outlier_bounds = load_regression_artifacts()
    if model is None:
        artifact_missing_warning("Regression")
    else:
        st.markdown('<p class="section-header">Tuned Random Forest — Test Performance</p>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3, gap="medium")
        for col, val, lbl in [
            (m1, "0.8922", "R² Score"),
            (m2, "£3,939.84", "Root Mean Squared Error (RMSE)"),
            (m3, "£1,521.90", "Mean Absolute Error (MAE)")
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="value">{val}</div><div class="label">{lbl}</div></div>',
                unsafe_allow_html=True)

        st.markdown('<p class="section-header">Configure Prediction Features</p>', unsafe_allow_html=True)

        with st.form("regression_form"):
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown("<p style='font-size:0.92rem;font-weight:600;color:#303247;margin:0 0 0.5rem;'>Order & Catalog Scope</p>", unsafe_allow_html=True)
                product_variation_count = st.number_input("Product Variation Count (unique SKUs)", 1, 1000, 15)
                total_items_sold = st.number_input("Total Items Sold (quantity)", 1, 100000, 300)
                total_orders = st.number_input("Total Orders (invoices)", 1, 10000, 20)
                avg_items_per_order = st.number_input("Avg Items per Order", 0.1, 500.0, 15.0, step=0.1)

            with col2:
                st.markdown("<p style='font-size:0.92rem;font-weight:600;color:#303247;margin:0 0 0.5rem;'>Calendar & Geographic Context</p>", unsafe_allow_html=True)
                country = st.selectbox("Country", [
                    "United Kingdom", "France", "Germany", "EIRE", "Netherlands",
                    "Spain", "Belgium", "Switzerland", "Portugal", "Australia",
                    "Norway", "Denmark", "Sweden", "Finland", "Austria", "Italy", "Other",
                ])
                year = st.number_input("Year", 2010, 2030, 2011)
                month = st.selectbox("Month", list(range(1, 13)), index=10,
                                     format_func=lambda m: pd.Timestamp(2011, m, 1).strftime("%B"))
                day_of_week = st.selectbox(
                    "Day of Week",
                    list(range(7)),
                    format_func=lambda d: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][d],
                )

            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Predict Daily Sales Volume", use_container_width=True)

        if submitted:
            row = pd.DataFrame([{
                "product_variation_count": float(product_variation_count),
                "total_items_sold": float(total_items_sold),
                "total_orders": float(total_orders),
                "year": float(year),
                "month": float(month),
                "day_of_week": float(day_of_week),
                "avg_items_per_order": float(avg_items_per_order),
                "Country": country,
            }])
            for feat in ["product_variation_count", "total_items_sold", "total_orders", "avg_items_per_order"]:
                if feat in outlier_bounds:
                    lb, ub = outlier_bounds[feat]
                    row[feat] = row[feat].clip(lb, ub)
            try:
                X_proc = preprocessor.transform(row)
                pred = float(np.expm1(model.predict(X_proc)[0]))
                pred = max(pred, 0.0)
                st.markdown(
                    f'<div class="result-card prediction-box">'
                    f'<div class="res-label">Predicted Daily Sales Volume</div>'
                    f'<div class="res-value pred-value">&#xa3;{pred:,.2f}</div>'
                    f'<div class="res-desc">'
                    f'{country} &bull; {pd.Timestamp(int(year), int(month), 1).strftime("%B")} &bull; '
                    f'{["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][int(day_of_week)]}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.markdown('<p class="section-header">All Regression Models — Evaluation Summary</p>', unsafe_allow_html=True)
        reg_table = pd.DataFrame([
            {"Model": "Linear Regression",              "R2": 0.8340, "RMSE": 4888.65, "MAE": 1983.81},
            {"Model": "Ridge Regression",               "R2": 0.8420, "RMSE": 4769.39, "MAE": 1980.41},
            {"Model": "Lasso Regression",               "R2": 0.8340, "RMSE": 4888.65, "MAE": 1983.81},
            {"Model": "ElasticNet Regression",          "R2": 0.8373, "RMSE": 4840.66, "MAE": 1977.25},
            {"Model": "Polynomial Regression (d=2)",    "R2": 0.8658, "RMSE": 4395.71, "MAE": 1748.68},
            {"Model": "Decision Tree Regressor",        "R2": 0.7711, "RMSE": 5740.90, "MAE": 2071.07},
            {"Model": "Random Forest (base)",           "R2": 0.8786, "RMSE": 4180.98, "MAE": 1564.38},
            {"Model": "Gradient Boosting Regressor",    "R2": 0.8632, "RMSE": 4438.79, "MAE": 1688.14},
            {"Model": "SVR (base)",                     "R2": 0.8964, "RMSE": 3861.66, "MAE": 1588.10},
            {"Model": "KNN Regressor",                  "R2": 0.8690, "RMSE": 4344.02, "MAE": 1823.61},
            {"Model": "Tuned SVR",                      "R2": 0.8427, "RMSE": 4758.58, "MAE": 1813.96},
            {"Model": "Tuned Random Forest (Selected)", "R2": 0.8922, "RMSE": 3939.84, "MAE": 1521.90},
        ]).sort_values("R2", ascending=False)
        st.dataframe(
            reg_table.style.format({"R2": "{:.4f}", "RMSE": "{:,.2f}", "MAE": "{:,.2f}"}),
            use_container_width=True, hide_index=True,
        )


# ════════════════════ CLASSIFICATION ═════════════════════════════════════════

elif page == "Classification":
    render_page_header(
        "Online Purchase Intent Prediction",
        "Predicts whether a browsing session concludes with a transaction (<b>Revenue = True</b>). "
        "Evaluated with an <b>SVM RBF kernel (C=10, gamma=scale)</b> trained with zero-leakage stratified scaling."
    )

    pre_data, model = load_classification_artifacts()
    if model is None:
        artifact_missing_warning("Classification")
    else:
        scaler = pre_data["scaler"]
        encoder = pre_data["encoder"]

        st.markdown('<p class="section-header">SVM (RBF, C=10) — Test Set Performance</p>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        for col, val, lbl in [
            (m1, "89.0%", "Overall Accuracy"),
            (m2, "0.883", "Weighted F1 Score"),
            (m3, "0.863", "ROC-AUC Score"),
            (m4, "69.6%", "Positive Precision"),
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="value">{val}</div><div class="label">{lbl}</div></div>',
                unsafe_allow_html=True)

        st.markdown('<p class="section-header">Session Behavioral Features</p>', unsafe_allow_html=True)

        month_options = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        visitor_options = ["Returning_Visitor", "New_Visitor", "Other"]

        with st.form("clf_form"):
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown("<p style='font-size:0.92rem;font-weight:600;color:#303247;margin:0 0 0.5rem;'>Page Visits & Engagement</p>", unsafe_allow_html=True)
                p1, p2, p3 = st.columns(3)
                admin = p1.number_input("Admin Pages", 0, 100, 0)
                info = p2.number_input("Info Pages", 0, 100, 0)
                prod = p3.number_input("Product Pages", 0, 1000, 5)

                d1, d2, d3 = st.columns(3)
                admin_dur = d1.number_input("Admin Time (s)", 0.0, 10000.0, 0.0, step=1.0)
                info_dur = d2.number_input("Info Time (s)", 0.0, 10000.0, 0.0, step=1.0)
                prod_dur = d3.number_input("Product Time (s)", 0.0, 50000.0, 120.0, step=1.0)

                e1, e2 = st.columns(2)
                bounce = e1.number_input("Bounce Rate", 0.0, 0.2, 0.02, step=0.001, format="%.4f")
                exit_rate = e2.number_input("Exit Rate", 0.0, 0.2, 0.03, step=0.001, format="%.4f")

                e3, e4 = st.columns(2)
                page_val = e3.number_input("Page Values", 0.0, 400.0, 0.0, step=0.1)
                special = e4.number_input("Special Day (0-1)", 0.0, 1.0, 0.0, step=0.1)

            with col2:
                st.markdown("<p style='font-size:0.92rem;font-weight:600;color:#303247;margin:0 0 0.5rem;'>Session Context & Technical Signals</p>", unsafe_allow_html=True)
                month = st.selectbox("Session Month", month_options)
                visitor_type = st.selectbox("Visitor Type", visitor_options)
                weekend = st.selectbox("Weekend Session?", ["False", "True"])

                t1, t2 = st.columns(2)
                os = t1.number_input("Operating System (1-8)", 1, 8, 2)
                browser = t2.number_input("Browser (1-13)", 1, 13, 2)

                t3, t4 = st.columns(2)
                region = t3.number_input("Region (1-9)", 1, 9, 1)
                traffic = t4.number_input("Traffic Type (1-20)", 1, 20, 2)

            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Predict Purchase Intent", use_container_width=True)

        if submitted:
            total_pages = admin + info + prod
            total_dur = admin_dur + info_dur + prod_dur
            avg_dur_per_page = total_dur / (total_pages + 1e-5)
            exit_to_bounce = exit_rate / (bounce + 1e-5)

            num_cols = [
                "Administrative", "Administrative_Duration",
                "Informational", "Informational_Duration",
                "ProductRelated", "ProductRelated_Duration",
                "BounceRates", "ExitRates", "PageValues", "SpecialDay",
                "OperatingSystems", "Browser", "Region", "TrafficType",
                "Total_Pages_Visited", "Total_Duration", "Avg_Duration_Per_Page", "Exit_to_Bounce_Ratio"
            ]
            cat_cols = ["Month", "VisitorType", "Weekend"]

            raw_num = pd.DataFrame([[
                admin, admin_dur, info, info_dur, prod, prod_dur,
                bounce, exit_rate, page_val, float(special),
                float(os), float(browser), float(region), float(traffic),
                total_pages, total_dur, avg_dur_per_page, exit_to_bounce,
            ]], columns=num_cols)
            raw_cat = pd.DataFrame([[month, visitor_type, (weekend == "True")]], columns=cat_cols)

            try:
                X_num = scaler.transform(raw_num)
                X_cat = encoder.transform(raw_cat)
                cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist()
                all_feature_names = num_cols + cat_feature_names
                X_full = pd.DataFrame(np.hstack([X_num, X_cat]), columns=all_feature_names)
                pred = int(model.predict(X_full)[0])

                if pred == 1:
                    st.markdown(
                        '<div class="result-card success prediction-box">'
                        '<div class="res-label" style="color:#28573d">Purchase Predicted</div>'
                        '<div class="res-value pred-value">Revenue = True</div>'
                        '<div class="res-desc">Session demonstrates high purchasing intent supported by page engagement and page values.</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-card warning prediction-box">'
                        '<div class="res-label" style="color:#8c4633">Non-Conversion Predicted</div>'
                        '<div class="res-value pred-value">Revenue = False</div>'
                        '<div class="res-desc">Session indicates exploratory browsing without immediate transaction indicators.</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.markdown('<p class="section-header">All Classifiers Comparison (Test Set)</p>', unsafe_allow_html=True)
        clf_table = pd.DataFrame([
            {"Algorithm": "Dummy Baseline",          "Accuracy": 0.8451, "F1 (Weighted)": 0.7741, "ROC-AUC": 0.5000, "Positive Recall": 0.000},
            {"Algorithm": "Logistic Regression",     "Accuracy": 0.8812, "F1 (Weighted)": 0.8630, "ROC-AUC": 0.8875, "Positive Recall": 0.356},
            {"Algorithm": "KNN (k=13, manhattan)",   "Accuracy": 0.8771, "F1 (Weighted)": 0.8575, "ROC-AUC": 0.8141, "Positive Recall": 0.335},
            {"Algorithm": "Gaussian Naive Bayes",    "Accuracy": 0.7936, "F1 (Weighted)": 0.8110, "ROC-AUC": 0.8026, "Positive Recall": 0.623},
            {"Algorithm": "Decision Tree (depth=5)", "Accuracy": 0.8974, "F1 (Weighted)": 0.8931, "ROC-AUC": 0.9177, "Positive Recall": 0.579},
            {"Algorithm": "SVM RBF C=5",             "Accuracy": 0.8890, "F1 (Weighted)": 0.8743, "ROC-AUC": 0.8698, "Positive Recall": 0.448},
            {"Algorithm": "SVM RBF C=10 (Selected)", "Accuracy": 0.8901, "F1 (Weighted)": 0.8832, "ROC-AUC": 0.8627, "Positive Recall": 0.500},
        ]).sort_values("F1 (Weighted)", ascending=False)
        st.dataframe(
            clf_table.style.format({
                "Accuracy": "{:.1%}", "F1 (Weighted)": "{:.4f}",
                "ROC-AUC": "{:.4f}", "Positive Recall": "{:.1%}",
            }),
            use_container_width=True, hide_index=True,
        )


# ════════════════════ CLUSTERING ══════════════════════════════════════════════

elif page == "Clustering":
    render_page_header(
        "E-Commerce Session Clustering",
        "Segmenting 9.24 million e-commerce user sessions into <b>4 distinct behavioral clusters</b> "
        "using <b>MiniBatch K-Means (K=4)</b> on session-level aggregated activity."
    )

    scaler, km_model, cluster_profile, cluster_summary = load_clustering_artifacts()

    st.markdown('<p class="section-header">Identified E-Commerce Session Segments</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown("""<div class="cluster-card" style="border-top:3.5px solid #8B83C9;">
            <div>
                <span class="badge badge-accent">CLUSTER 0 &bull; 15.5%</span>
                <h4>High-Value Browsers</h4>
                <p>High viewed item prices (&pound;963 avg), 3.6 views, zero carts. Luxury / high-ticket interest without immediate commitment.</p>
            </div>
            <p style="margin-top:0.6rem;padding-top:0.4rem;border-top:1px solid #E6E4EF;font-size:0.78rem;color:#73758A;">
                <b>Focus:</b> Premium Window Shopping
            </p>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class="cluster-card" style="border-top:3.5px solid #A9B9E8;">
            <div>
                <span class="badge badge-regression">CLUSTER 1 &bull; 73.2%</span>
                <h4>Casual Browsers</h4>
                <p>Low price viewed (&pound;181 avg), ~3 page views, zero carts. Represents standard fast window shoppers.</p>
            </div>
            <p style="margin-top:0.6rem;padding-top:0.4rem;border-top:1px solid #E6E4EF;font-size:0.78rem;color:#73758A;">
                <b>Focus:</b> Quick Casual Visits
            </p>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""<div class="cluster-card" style="border-top:3.5px solid #A8C9B5;">
            <div>
                <span class="badge badge-classification">CLUSTER 2 &bull; 5.3%</span>
                <h4>Catalog Researchers</h4>
                <p>Extensive depth (15.6 unique SKUs, 23.4 views, 37.4 min duration). High comparative research before buying.</p>
            </div>
            <p style="margin-top:0.6rem;padding-top:0.4rem;border-top:1px solid #E6E4EF;font-size:0.78rem;color:#73758A;">
                <b>Focus:</b> Deep In-Catalog Evaluation
            </p>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown("""<div class="cluster-card" style="border-top:3.5px solid #E8B9A8;">
            <div>
                <span class="badge badge-clustering">CLUSTER 3 &bull; 6.0%</span>
                <h4>Carting Shoppers</h4>
                <p>Active cart additions (1.58 avg, 1.10 unique products). High-intent shopping sessions closest to conversion.</p>
            </div>
            <p style="margin-top:0.6rem;padding-top:0.4rem;border-top:1px solid #E6E4EF;font-size:0.78rem;color:#73758A;">
                <b>Focus:</b> Direct Pre-Purchase Intent
            </p>
        </div>""", unsafe_allow_html=True)

    if km_model is None:
        artifact_missing_warning("Clustering")

    st.markdown('<p class="section-header">MiniBatch K-Means (K=4) — Clustering Validation</p>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3, gap="medium")
    for col, val, lbl in [
        (m1, "0.542", "Silhouette Score (50k sample)"),
        (m2, "0.914", "Davies-Bouldin Index (lower=better)"),
        (m3, "15,991", "Calinski-Harabasz Index (higher=better)"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="value">{val}</div><div class="label">{lbl}</div></div>',
            unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2], gap="large")
    with col_a:
        st.markdown("<p style='font-size:0.92rem;font-weight:600;color:#303247;margin:1rem 0 0.4rem;'>Session Distribution by Cluster</p>", unsafe_allow_html=True)
        dist_df = pd.DataFrame({
            "Cluster": [0, 1, 2, 3],
            "Segment": ["High-Value Browsers", "Casual Browsers", "Catalog Researchers", "Carting Shoppers"],
            "Sessions": [1431933, 6770409, 486228, 555851],
            "Percentage (%)": [15.49, 73.24, 5.26, 6.01]
        })
        st.dataframe(
            dist_df.style.format({"Sessions": "{:,.0f}", "Percentage (%)": "{:.2f}%"}),
            use_container_width=True, hide_index=True
        )

    with col_b:
        st.markdown("<p style='font-size:0.92rem;font-weight:600;color:#303247;margin:1rem 0 0.4rem;'>Cluster Feature Averages Profile</p>", unsafe_allow_html=True)
        prof_df = pd.DataFrame([
            {"Cluster": 0, "Products": 2.70, "Views": 3.62, "Carts": 0.00, "Avg Price (£)": 963.20, "Duration (min)": 16.67},
            {"Cluster": 1, "Products": 2.20, "Views": 3.14, "Carts": 0.00, "Avg Price (£)": 180.54, "Duration (min)": 16.54},
            {"Cluster": 2, "Products": 15.59, "Views": 23.42, "Carts": 0.03, "Avg Price (£)": 232.62, "Duration (min)": 37.43},
            {"Cluster": 3, "Products": 2.65, "Views": 5.31, "Carts": 1.58, "Avg Price (£)": 332.88, "Duration (min)": 11.62},
        ])
        st.dataframe(
            prof_df.style.format({
                "Products": "{:.2f}", "Views": "{:.2f}", "Carts": "{:.2f}",
                "Avg Price (£)": "{:,.2f}", "Duration (min)": "{:.2f}"
            }),
            use_container_width=True, hide_index=True
        )

    if km_model is not None:
        st.markdown('<p class="section-header">Assign a New Session to a Segment</p>', unsafe_allow_html=True)

        with st.form("cluster_form"):
            c1, c2 = st.columns(2, gap="large")
            with c1:
                uniq_prod = c1.number_input("Unique Products Viewed", 0, 500, 3)
                uniq_cat = c2.number_input("Unique Categories Viewed", 0, 100, 1)
                total_views = c1.number_input("Total Page Views", 0, 1000, 4)
                cart_add = c2.number_input("Cart Additions", 0, 200, 0)
            with c2:
                uniq_carted = c1.number_input("Unique Products Carted", 0, 200, 0)
                avg_price = c2.number_input("Avg Price Viewed (£)", 0.0, 5000.0, 50.0, step=0.5)
                max_price = c1.number_input("Max Price Viewed (£)", 0.0, 10000.0, 100.0, step=0.5)
                session_dur = c2.number_input("Session Duration (minutes)", 0.0, 600.0, 5.0, step=0.5)

            submitted = st.form_submit_button("Assign to Cluster", use_container_width=True)

        if submitted:
            row = np.array([[uniq_prod, uniq_cat, total_views, cart_add, uniq_carted,
                             avg_price, max_price, session_dur]], dtype=float)
            try:
                row_scaled = scaler.transform(row)
                cluster_id = int(km_model.predict(row_scaled)[0])
                seg_names = {0: "High-Value Browsers", 1: "Casual Browsers", 2: "Catalog Researchers", 3: "Carting Shoppers"}
                st.markdown(
                    f'<div class="result-card">'
                    f'<div class="res-label">Assigned Cluster</div>'
                    f'<div class="res-value">Cluster {cluster_id}: {seg_names.get(cluster_id, "")}</div>'
                    f'<div class="res-desc">Assigned by MiniBatch K-Means based on session engagement vector.</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Assignment failed: {e}")

    st.markdown('<p class="section-header">Cluster Count Sweep — Metric Comparison</p>', unsafe_allow_html=True)
    sweep_df = pd.DataFrame({
        "K": [2, 3, 4, 5, 6],
        "Silhouette": [0.608, 0.532, 0.542, 0.527, 0.402],
        "Davies-Bouldin": [0.861, 1.184, 0.914, 1.031, 1.043],
        "Calinski-Harabasz": [11053, 13435, 15991, 14528, 12000],
        "Optimal": ["", "", "K=4 (Selected)", "", ""],
    })
    st.dataframe(
        sweep_df.style.format({
            "Silhouette": "{:.3f}",
            "Davies-Bouldin": "{:.3f}",
            "Calinski-Harabasz": "{:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )


# ════════════════════ MODEL COMPARISON ════════════════════════════════════════

elif page == "Model Comparison":
    render_page_header(
        "Model Comparison & Benchmarks",
        "Comprehensive performance analysis across Regression, Classification, and Clustering algorithms. "
        "Metrics reflect held-out test splits with zero data leakage."
    )

    configure_matplotlib_theme()
    tab1, tab2, tab3 = st.tabs(["Regression Benchmarks", "Classification Benchmarks", "Clustering Metrics"])

    with tab1:
        st.markdown("<p style='font-weight:600;font-size:1.05rem;color:#303247;margin:0.25rem 0 0.1rem'>Regression Models (Target: Daily Sales Volume &pound;)</p>", unsafe_allow_html=True)
        st.caption("Inverse-transformed from log1p target. Evaluated on 20% stratified test set.")
        reg_df = pd.DataFrame([
            {"Model": "Linear Regression",              "R2": 0.8340, "RMSE": 4888.65, "MAE": 1983.81},
            {"Model": "Ridge Regression",               "R2": 0.8420, "RMSE": 4769.39, "MAE": 1980.41},
            {"Model": "Lasso Regression",               "R2": 0.8340, "RMSE": 4888.65, "MAE": 1983.81},
            {"Model": "ElasticNet Regression",          "R2": 0.8373, "RMSE": 4840.66, "MAE": 1977.25},
            {"Model": "Polynomial Regression (d=2)",    "R2": 0.8658, "RMSE": 4395.71, "MAE": 1748.68},
            {"Model": "Decision Tree Regressor",        "R2": 0.7711, "RMSE": 5740.90, "MAE": 2071.07},
            {"Model": "Random Forest (base)",           "R2": 0.8786, "RMSE": 4180.98, "MAE": 1564.38},
            {"Model": "Gradient Boosting Regressor",    "R2": 0.8632, "RMSE": 4438.79, "MAE": 1688.14},
            {"Model": "SVR (base, C=1)",                "R2": 0.8964, "RMSE": 3861.66, "MAE": 1588.10},
            {"Model": "KNN Regressor",                  "R2": 0.8690, "RMSE": 4344.02, "MAE": 1823.61},
            {"Model": "Tuned SVR (C=100, gamma=0.01)",  "R2": 0.8427, "RMSE": 4758.58, "MAE": 1813.96},
            {"Model": "Tuned Random Forest [Selected]", "R2": 0.8922, "RMSE": 3939.84, "MAE": 1521.90},
        ]).sort_values("R2", ascending=False)

        st.dataframe(
            reg_df.style.format({"R2": "{:.4f}", "RMSE": "{:,.2f}", "MAE": "{:,.2f}"}),
            use_container_width=True, hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(9, 4.8))
        bar_colors = ["#8B83C9" if "Selected" in m else "#A9B9E8" for m in reg_df["Model"]]
        ax.barh(reg_df["Model"], reg_df["R2"], color=bar_colors, height=0.62)
        ax.set_xlabel("R² Score (Higher is Better)")
        ax.set_title("Regression Models — R² Score Comparison")
        ax.set_xlim(0.74, 0.92)
        ax.axvline(0.85, linestyle="--", color="#73758A", linewidth=0.9, label="0.85 Benchmark")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        st.markdown("<p style='font-weight:600;font-size:1.05rem;color:#303247;margin:0.25rem 0 0.1rem'>Classification Models (Target: Purchase Intent / Revenue)</p>", unsafe_allow_html=True)
        st.caption("Evaluated on 2,466 test sessions (20% stratified split). Baseline dummy accuracy is 84.51% with zero buyer capture.")
        clf_df = pd.DataFrame([
            {"Algorithm": "Dummy Baseline",          "Accuracy": 0.8451, "F1 (Weighted)": 0.7741, "ROC-AUC": 0.5000, "Positive Recall": 0.000},
            {"Algorithm": "Logistic Regression",     "Accuracy": 0.8812, "F1 (Weighted)": 0.8630, "ROC-AUC": 0.8875, "Positive Recall": 0.356},
            {"Algorithm": "KNN (k=13, manhattan)",   "Accuracy": 0.8771, "F1 (Weighted)": 0.8575, "ROC-AUC": 0.8141, "Positive Recall": 0.335},
            {"Algorithm": "Gaussian Naive Bayes",    "Accuracy": 0.7936, "F1 (Weighted)": 0.8110, "ROC-AUC": 0.8026, "Positive Recall": 0.623},
            {"Algorithm": "Decision Tree (depth=5)", "Accuracy": 0.8974, "F1 (Weighted)": 0.8931, "ROC-AUC": 0.9177, "Positive Recall": 0.579},
            {"Algorithm": "SVM RBF C=5",             "Accuracy": 0.8890, "F1 (Weighted)": 0.8743, "ROC-AUC": 0.8698, "Positive Recall": 0.448},
            {"Algorithm": "SVM RBF C=10 [Selected]", "Accuracy": 0.8901, "F1 (Weighted)": 0.8832, "ROC-AUC": 0.8627, "Positive Recall": 0.500},
        ]).sort_values("F1 (Weighted)", ascending=False)

        st.dataframe(
            clf_df.style.format({
                "Accuracy": "{:.1%}", "F1 (Weighted)": "{:.4f}",
                "ROC-AUC": "{:.4f}", "Positive Recall": "{:.1%}",
            }),
            use_container_width=True, hide_index=True,
        )

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        no_dummy = clf_df[clf_df["Algorithm"] != "Dummy Baseline"].copy()
        colors = ["#8B83C9" if "Selected" in a else "#A9B9E8" for a in no_dummy["Algorithm"]]
        axes[0].barh(no_dummy["Algorithm"], no_dummy["F1 (Weighted)"], color=colors, height=0.58)
        axes[0].set_xlabel("Weighted F1 Score")
        axes[0].set_title("Weighted F1 Comparison")
        axes[0].grid(axis='x', linestyle='--', alpha=0.7)
        axes[0].set_axisbelow(True)

        axes[1].barh(no_dummy["Algorithm"], no_dummy["ROC-AUC"], color=colors, height=0.58)
        axes[1].set_xlabel("ROC-AUC Score")
        axes[1].set_title("ROC-AUC Comparison")
        axes[1].grid(axis='x', linestyle='--', alpha=0.7)
        axes[1].set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab3:
        st.markdown("<p style='font-weight:600;font-size:1.05rem;color:#303247;margin:0.25rem 0 0.1rem'>Clustering Validation (MiniBatch K-Means)</p>", unsafe_allow_html=True)
        st.caption("Metric evaluation on 50,000-session stratified sample across K=2 through K=6.")
        clust_df = pd.DataFrame({
            "K": [2, 3, 4, 5, 6],
            "Silhouette (higher=better)": [0.608, 0.532, 0.542, 0.527, 0.402],
            "Davies-Bouldin (lower=better)": [0.861, 1.184, 0.914, 1.031, 1.043],
            "Calinski-Harabasz (higher=better)": [11053, 13435, 15991, 14528, 12000],
            "Selection Note": ["", "", "K=4 (Selected)", "", ""],
        })
        st.dataframe(
            clust_df.style.format({
                "Silhouette (higher=better)": "{:.3f}",
                "Davies-Bouldin (lower=better)": "{:.3f}",
                "Calinski-Harabasz (higher=better)": "{:,.0f}",
            }),
            use_container_width=True, hide_index=True,
        )

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        k_vals = clust_df["K"]
        axes[0].plot(k_vals, clust_df["Silhouette (higher=better)"], "o-", color="#8B83C9", linewidth=2.0, markersize=6)
        axes[0].axvline(4, linestyle="--", color="#A8C9B5", label="K=4 Selected", linewidth=1.5)
        axes[0].set_xlabel("Number of Clusters (K)")
        axes[0].set_ylabel("Silhouette Score")
        axes[0].set_title("Silhouette Score vs K")
        axes[0].set_xticks([2, 3, 4, 5, 6])
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].set_axisbelow(True)
        axes[0].legend(loc="upper right")

        axes[1].plot(k_vals, clust_df["Calinski-Harabasz (higher=better)"], "s-", color="#E8B9A8", linewidth=2.0, markersize=6)
        axes[1].axvline(4, linestyle="--", color="#A8C9B5", label="K=4 Selected", linewidth=1.5)
        axes[1].set_xlabel("Number of Clusters (K)")
        axes[1].set_ylabel("Calinski-Harabasz Index")
        axes[1].set_title("Calinski-Harabasz Index vs K")
        axes[1].set_xticks([2, 3, 4, 5, 6])
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].set_axisbelow(True)
        axes[1].legend(loc="upper right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ════════════════════ ABOUT ════════════════════════════════════════════════════

elif page == "About":
    render_page_header(
        "About This Project",
        "Course: 23CSE301 Machine Learning Capstone &bull; End-to-End E-Commerce Predictive & Clustering Architecture"
    )

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.markdown("""
<div class="custom-card">
    <h3 style="margin-top:0;">Project Objectives</h3>
    <p>This capstone investigates three distinct predictive challenges in modern e-commerce using real-world transactional and behavioral data:</p>
    <ul style="color:#73758A;font-size:0.9rem;line-height:1.65;margin-bottom:0;">
        <li><b>Track 1 (Regression):</b> Predict daily aggregate sales volume in GBP (£) from order frequency, SKU breadth, and calendar signals to assist inventory management.</li>
        <li><b>Track 2 (Classification):</b> Identify session conversion likelihood (Revenue = True) to enable automated real-time discount incentives without wasting margin.</li>
        <li><b>Track 3 (Clustering):</b> Segment multi-million session catalogs into actionable user groups to distinguish window browsing, deep research, and high-intent buying.</li>
    </ul>
</div>

<div class="custom-card">
    <h3 style="margin-top:0;">Datasets & Scope</h3>
    <table style="width:100%;font-size:0.88rem;color:#303247;border-collapse:collapse;">
        <tr style="border-bottom:1px solid #E6E4EF;text-align:left;color:#73758A;">
            <th style="padding:7px 0;">Dataset</th>
            <th style="padding:7px 0;">Dimensions</th>
            <th style="padding:7px 0;">ML Domain</th>
        </tr>
        <tr style="border-bottom:1px solid #E6E4EF;">
            <td style="padding:9px 0;">Online Retail (UK)</td>
            <td style="padding:9px 0;">541,909 rows &rarr; 1,554 daily aggregates</td>
            <td style="padding:9px 0;"><span class="badge badge-regression">Regression</span></td>
        </tr>
        <tr style="border-bottom:1px solid #E6E4EF;">
            <td style="padding:9px 0;">Online Shoppers Intention</td>
            <td style="padding:9px 0;">12,330 sessions &bull; 30 features</td>
            <td style="padding:9px 0;"><span class="badge badge-classification">Classification</span></td>
        </tr>
        <tr>
            <td style="padding:9px 0;">E-Commerce Events (Oct 2019)</td>
            <td style="padding:9px 0;">~42M events &rarr; 9.24M user sessions</td>
            <td style="padding:9px 0;"><span class="badge badge-clustering">Clustering</span></td>
        </tr>
    </table>
</div>

<div class="custom-card">
    <h3 style="margin-top:0;">Data Preprocessing & Leakage Prevention</h3>
    <p style="font-size:0.88rem;color:#73758A;line-height:1.65;margin:0;">
        <b>Zero Data Leakage:</b> All scaling parameters (mean, standard deviation) and categorical encoder bins are fitted strictly on the 80% training splits and subsequently applied to test sets.<br>
        <b>Outlier Clipping:</b> Interquartile Range (IQR) bounds [Q1 - 1.5&times;IQR, Q3 + 1.5&times;IQR] are derived from training features to stabilize regressor performance against extreme order spikes.<br>
        <b>Feature Engineering:</b> 4 interaction metrics created for classification: <i>Total_Pages_Visited</i>, <i>Total_Duration</i>, <i>Avg_Duration_Per_Page</i>, and <i>Exit_to_Bounce_Ratio</i>.
    </p>
</div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="custom-card">
    <h3 style="margin-top:0;">Environment</h3>
    <p style="font-size:0.88rem;color:#73758A;line-height:1.75;margin:0;">
        &bull; <b>Python:</b> 3.12<br>
        &bull; <b>Streamlit:</b> 1.65+<br>
        &bull; <b>scikit-learn:</b> 1.8<br>
        &bull; <b>pandas:</b> 2.2<br>
        &bull; <b>numpy:</b> 2.0<br>
        &bull; <b>matplotlib:</b> 3.9<br>
        &bull; <b>joblib:</b> 1.5
    </p>
</div>
""", unsafe_allow_html=True)

        st.markdown('<p class="section-header" style="margin-top:0.5rem">Start Application</p>', unsafe_allow_html=True)
        st.code("streamlit run app.py", language="bash")
        st.markdown('<p class="section-header" style="margin-top:0.5rem">Export Models</p>', unsafe_allow_html=True)
        st.code("python export_models.py --task regression\npython export_models.py --task classification", language="bash")
