import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent / "notebooks"))

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

:root {
    /* Palette */
    --app-bg: #F7F6FC;
    --app-sidebar: #E5EAF8;
    --app-surface: #FFFFFF;
    --app-border: #E6E4EF;
    --app-border-strong: #D8D6E5;
    --app-accent: #8B83C9;
    --app-accent-hover: #7b73b8;
    --app-blue: #A9B9E8;
    --app-sage: #A8C9B5;
    --app-peach: #E8B9A8;
    --app-text: #303247;
    --app-text-2: #626478;

    /* Radii */
    --app-radius-sm: 8px;
    --app-radius-md: 10px;
    --app-radius-lg: 12px;
    --app-radius-pill: 999px;

    /* Shadows */
    --app-shadow-sm: 0 1px 3px rgba(48, 50, 71, 0.03);
    --app-shadow-accent: 0 2px 8px rgba(139, 131, 201, 0.08);

    /* Spacing */
    --app-space-1: 0.25rem;
    --app-space-2: 0.5rem;
    --app-space-3: 0.75rem;
    --app-space-4: 1rem;
    --app-space-5: 1.25rem;
    --app-space-6: 1.5rem;
    --app-space-8: 2rem;

    /* Typography */
    --app-fs-xs: 0.75rem;
    --app-fs-sm: 0.82rem;
    --app-fs-base: 0.9rem;
    --app-fs-md: 1rem;
    --app-fs-lg: 1.15rem;
    --app-fs-xl: 1.85rem;
}

*, *::before, *::after {
    box-sizing: border-box;
}

.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--app-text);
    background-color: var(--app-bg);
}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: var(--app-text);
    font-weight: 600;
    letter-spacing: -0.015em;
    line-height: 1.3;
}

header[data-testid="stHeader"] {
    background-color: transparent !important;
}

.stApp [data-testid="stMainBlockContainer"] {
    width: 100%;
    max-width: 1440px;
    padding: 1.75rem clamp(1rem, 3vw, 2.5rem) 3rem;
}

[data-testid="stSidebar"] {
    background-color: var(--app-sidebar);
    border-right: 1px solid var(--app-border);
}

.stApp [data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 0.45rem 0.75rem;
    border-radius: var(--app-radius-sm);
    transition: background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease;
    margin: 0 0 0.2rem;
    cursor: pointer;
    line-height: 1.4;
    border-left: 3px solid transparent;
}

.stApp [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: rgba(139, 131, 201, 0.12);
}

.stApp [data-testid="stSidebar"] div[role="radiogroup"] > label:focus-within {
    outline: 2px solid var(--app-accent);
    outline-offset: 1px;
}

.stApp [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background-color: rgba(139, 131, 201, 0.18);
    color: var(--app-text);
    font-weight: 600;
    border-left: 3px solid var(--app-accent);
}

.stApp [data-testid="stMarkdownContainer"] p {
    line-height: 1.55;
}

.stApp [data-testid="column"] {
    display: flex;
    flex-direction: column;
}

.stApp [data-testid="column"] > div {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
    width: 100%;
}

.stApp [data-testid="column"] .metric-card,
.stApp [data-testid="column"] .cluster-card,
.stApp [data-testid="column"] .custom-card {
    flex: 1 1 auto;
    height: 100%;
    margin-bottom: 0;
}

.stApp [data-testid="column"] .hero-card {
    flex: 1 1 auto;
    height: 100%;
    margin-bottom: 0.75rem;
}

.stApp .custom-card,
.stApp .metric-card,
.stApp .hero-card,
.stApp .cluster-card,
.stApp .result-card {
    background-color: var(--app-surface);
    border: 1px solid var(--app-border);
    border-radius: var(--app-radius-lg);
    box-shadow: var(--app-shadow-sm);
    min-width: 0;
    overflow-wrap: anywhere;
}

.stApp .custom-card {
    padding: 1.35rem 1.5rem;
    margin-bottom: var(--app-space-4);
}

.stApp .metric-card {
    padding: var(--app-space-4) 0.85rem;
    text-align: center;
    min-height: 108px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 0.35rem;
}

.stApp .metric-card .value {
    font-size: clamp(1.15rem, 2vw, 1.5rem);
    font-weight: 700;
    color: var(--app-accent);
    line-height: 1.3;
    white-space: normal;
    overflow-wrap: anywhere;
    max-width: 100%;
}

.stApp .metric-card .label {
    font-size: var(--app-fs-sm);
    font-weight: 500;
    color: var(--app-text-2);
    line-height: 1.4;
    width: 100%;
}

.stApp .hero-card {
    padding: 1.35rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 17rem;
    height: calc(100% - 0.75rem);
    margin-bottom: 0.75rem;
    gap: 1rem;
}

.stApp .hero-card h3 {
    font-size: var(--app-fs-lg);
    margin: 0.45rem 0 0.35rem;
    color: var(--app-text);
}

.stApp .hero-card p {
    color: var(--app-text-2);
    font-size: 0.88rem;
    line-height: 1.55;
    margin: 0 0 0.4rem;
}

.stApp .cluster-card {
    padding: var(--app-space-5);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 13rem;
    height: calc(100% - 0.75rem);
    margin-bottom: var(--app-space-3);
    gap: var(--app-space-4);
}

.stApp .cluster-card h4 {
    margin: 0.45rem 0 0.3rem;
    font-size: var(--app-fs-md);
    color: var(--app-text);
}

.stApp .cluster-card p {
    font-size: 0.85rem;
    color: var(--app-text-2);
    line-height: 1.5;
    margin: 0;
}

.stApp .badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: var(--app-radius-pill);
    font-size: var(--app-fs-xs);
    font-weight: 600;
    letter-spacing: 0.02em;
    width: fit-content;
    line-height: 1.4;
}

.stApp .badge-regression {
    background-color: rgba(169, 185, 232, 0.3);
    color: #40507a;
}

.stApp .badge-classification {
    background-color: rgba(168, 201, 181, 0.35);
    color: #2b553e;
}

.stApp .badge-clustering {
    background-color: rgba(232, 185, 168, 0.45);
    color: #7b4231;
}

.stApp .badge-accent {
    background-color: rgba(139, 131, 201, 0.25);
    color: #554d8a;
}

.stApp .section-header {
    font-size: var(--app-fs-md);
    font-weight: 600;
    color: var(--app-text);
    border-left: 3px solid var(--app-accent);
    padding-left: 0.65rem;
    margin: 1.5rem 0 0.85rem;
    line-height: 1.35;
}

.stApp .result-card {
    border: 1.5px solid var(--app-accent);
    padding: var(--app-space-6) var(--app-space-8);
    text-align: center;
    box-shadow: var(--app-shadow-accent);
    margin: var(--app-space-5) 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.25rem;
}

.stApp .result-card.success {
    background-color: rgba(168, 201, 181, 0.18);
    border-color: var(--app-sage);
}

.stApp .result-card.warning {
    background-color: rgba(232, 185, 168, 0.22);
    border-color: var(--app-peach);
}

.stApp .result-card .res-label {
    font-size: var(--app-fs-sm);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--app-text-2);
}

.stApp .result-card .res-value {
    font-size: clamp(1.55rem, 4vw, 2.15rem);
    font-weight: 700;
    color: var(--app-text);
    margin: 0.35rem 0;
    line-height: 1.25;
    overflow-wrap: anywhere;
}

.stApp .result-card.success .res-value {
    color: #28573d;
}

.stApp .result-card.warning .res-value {
    color: #8c4633;
}

.stApp .result-card .res-desc {
    font-size: 0.88rem;
    color: var(--app-text-2);
    margin-top: 0.1rem;
    line-height: 1.5;
}

.stApp .pastel-notice {
    background-color: rgba(232, 185, 168, 0.12);
    border: 1px solid rgba(232, 185, 168, 0.4);
    border-radius: var(--app-radius-md);
    padding: 0.85rem 1.15rem;
    color: var(--app-text-2);
    margin: 0.5rem 0 1.5rem 0;
    line-height: 1.55;
    font-size: 0.85rem;
    overflow-wrap: anywhere;
}

.stApp .pastel-notice b {
    color: var(--app-text);
    font-weight: 600;
}

.stApp .pastel-notice code {
    background-color: rgba(255, 255, 255, 0.8);
    color: #8c4633;
    padding: 0.12rem 0.4rem;
    border-radius: 4px;
    border: 1px solid rgba(232, 185, 168, 0.3);
    font-size: 0.8rem;
    font-weight: 500;
    overflow-wrap: anywhere;
}

.stApp [data-testid="stButton"] > button,
.stApp [data-testid="stFormSubmitButton"] > button {
    min-height: 2.65rem;
    width: 100%;
    border-radius: var(--app-radius-md);
    font-weight: 500;
    border: 1px solid var(--app-border);
    background-color: var(--app-surface);
    color: var(--app-text);
    transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
    padding: 0.55rem 1rem;
    white-space: normal;
    line-height: 1.35;
}

.stApp [data-testid="stButton"] > button:focus-visible,
.stApp [data-testid="stFormSubmitButton"] > button:focus-visible {
    outline: 2px solid var(--app-accent);
    outline-offset: 2px;
}

.stApp [data-testid="stButton"] > button:hover {
    border-color: var(--app-accent);
    color: var(--app-accent);
    background-color: var(--app-bg);
}

.stApp [data-testid="stFormSubmitButton"] > button,
.stApp [data-testid="stButton"] > button[kind="primary"] {
    background-color: var(--app-accent);
    color: #FFFFFF;
    border: 1px solid var(--app-accent);
    font-weight: 600;
    border-radius: var(--app-radius-md);
}

.stApp [data-testid="stFormSubmitButton"] > button:hover,
.stApp [data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: var(--app-accent-hover);
    border-color: var(--app-accent-hover);
    color: #FFFFFF;
}

.stApp [data-testid="stForm"] {
    background-color: var(--app-surface);
    border: 1px solid var(--app-border);
    border-radius: var(--app-radius-lg);
    padding: 1.35rem;
    box-shadow: var(--app-shadow-sm);
}

.stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="select"] > div {
    border-radius: var(--app-radius-sm);
    border-color: var(--app-border-strong);
}

.stApp [data-baseweb="input"]:focus-within > div,
.stApp [data-baseweb="select"]:focus-within > div {
    border-color: var(--app-accent);
    box-shadow: 0 0 0 1px var(--app-accent);
}

.stApp [data-testid="stDataFrame"],
.stApp [data-testid="stTable"] {
    width: 100% !important;
    max-width: 100%;
    border: 1px solid var(--app-border);
    border-radius: var(--app-radius-lg);
    background-color: var(--app-surface);
    overflow: hidden;
    margin-bottom: var(--app-space-4);
}

.stApp [data-testid="stMainBlockContainer"] [data-testid="stRadio"] {
    background-color: var(--app-surface);
    border: 1px solid var(--app-border);
    border-radius: var(--app-radius-lg);
    padding: 0.85rem 1.25rem;
    box-shadow: var(--app-shadow-sm);
    margin: 0.25rem 0 1rem;
}

.stApp [data-testid="stMainBlockContainer"] [data-testid="stRadio"] label[data-testid="stWidgetLabel"] p {
    font-size: var(--app-fs-xs);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--app-text-2);
    margin: 0 0 0.45rem 0;
}

.stApp [data-testid="stMainBlockContainer"] [data-testid="stRadio"] div[role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    align-items: center;
}

.stApp [data-testid="stMainBlockContainer"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    padding: 0.45rem 1rem;
    border-radius: var(--app-radius-md);
    background-color: var(--app-bg);
    border: 1px solid var(--app-border);
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0;
}

.stApp [data-testid="stMainBlockContainer"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    border-color: var(--app-accent);
    background-color: rgba(139, 131, 201, 0.08);
}

.stApp [data-testid="stMainBlockContainer"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background-color: rgba(139, 131, 201, 0.18);
    border-color: var(--app-accent);
    color: var(--app-text);
    font-weight: 600;
}

.stApp [data-testid="stImage"] {
    display: flex;
    justify-content: center;
    width: 100%;
    margin: 0.5rem 0;
}

.stApp [data-testid="stImage"] img {
    max-height: 520px;
    object-fit: contain;
    border-radius: var(--app-radius-lg);
    border: 1px solid var(--app-border);
    box-shadow: var(--app-shadow-sm);
    background-color: #FFFFFF;
}

.stApp [data-testid="stImage"] [data-testid="stImageCaption"] {
    text-align: center;
    color: var(--app-text-2);
    font-size: var(--app-fs-sm);
    margin-top: 0.4rem;
}

.stApp [data-testid="stCodeBlock"] {
    border: 1px solid var(--app-border);
    border-radius: var(--app-radius-md);
    overflow-x: auto;
}

.stApp [data-testid="stTabs"] [role="tablist"] {
    gap: 0.5rem;
    overflow-x: auto;
    border-bottom: 1px solid var(--app-border);
    padding-bottom: 0.25rem;
    margin-bottom: 1rem;
}

.stApp [data-testid="stTabs"] button[role="tab"] {
    border-radius: var(--app-radius-sm);
    font-weight: 500;
    font-size: var(--app-fs-sm);
    color: var(--app-text-2);
    padding: 0.45rem 0.9rem;
    transition: all 0.15s ease;
}

.stApp [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--app-accent);
    font-weight: 600;
}

.stApp [data-testid="stMarkdownContainer"] table {
    width: 100%;
    max-width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    line-height: 1.45;
}

.stApp [data-testid="stMarkdownContainer"] th,
.stApp [data-testid="stMarkdownContainer"] td {
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid #E6E4EF;
    text-align: left;
    vertical-align: top;
}

.stApp [data-testid="stMarkdownContainer"] th {
    font-weight: 600;
    color: #303247;
}

.stApp hr {
    border-color: #E6E4EF;
    margin: 1.25rem 0;
}

.stApp .home-metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
}

@media (max-width: 900px) {
    .stApp [data-testid="stMainBlockContainer"] {
        padding: 1.5rem 1.25rem 2.5rem;
    }

    .stApp .home-metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.85rem;
    }
}

@media (max-width: 560px) {
    .stApp [data-testid="stMainBlockContainer"] {
        padding: 1.1rem 0.85rem 2rem;
    }

    .stApp .home-metrics {
        gap: 0.65rem;
    }

    .stApp .custom-card,
    .stApp .hero-card,
    .stApp .cluster-card,
    .stApp .result-card,
    .stApp [data-testid="stForm"] {
        padding: 1rem;
    }

    .stApp .hero-card,
    .stApp .cluster-card {
        min-height: 0;
    }

    .stApp .result-card .res-value {
        font-size: clamp(1.35rem, 7vw, 1.8rem);
    }

    .stApp [data-testid="stMarkdownContainer"] table {
        display: block;
        overflow-x: auto;
    }
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
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'axes.titlecolor': '#303247',
        'axes.titlepad': 10,
        'xtick.color': '#626478',
        'ytick.color': '#626478',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'grid.color': '#F2F1F8',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.8,
        'text.color': '#303247',
        'font.sans-serif': ['Inter', 'Segoe UI', 'DejaVu Sans', 'sans-serif'],
    })

configure_matplotlib_theme()


def render_page_header(title: str, subtitle: str):
    st.markdown(
        f'<div style="margin-bottom: 0.5rem;">'
        f'<h1 style="margin: 0 0 0.35rem 0; font-size: 1.85rem; font-weight: 700; color: #303247; letter-spacing: -0.02em;">{title}</h1>'
        f'<p style="margin: 0; color: #626478; font-size: 0.95rem; line-height: 1.5;">{subtitle}</p>'
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
    """Load the default classification model (SVM) and preprocessor for backward compatibility."""
    try:
        pre = joblib.load(MODELS_DIR / "classification_preprocessor.pkl")
        mdl = joblib.load(MODELS_DIR / "classification_model_svm.pkl")
        return pre, mdl
    except FileNotFoundError:
        return None, None


@st.cache_resource(show_spinner=False)
def load_all_classification_models():
    """Load all 10 classification models and the shared preprocessor."""
    try:
        pre = joblib.load(MODELS_DIR / "classification_preprocessor.pkl")
        model_names = ["logistic", "knn", "nb", "dt", "svm", "rf", "ada", "gb", "bag", "mlp"]
        models = {}
        for name in model_names:
            path = MODELS_DIR / f"classification_model_{name}.pkl"
            if path.exists():
                models[name] = joblib.load(path)
        return pre, models
    except FileNotFoundError:
        return None, {}


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


@st.cache_resource(show_spinner=False)
def load_agg_clustering_artifacts():
    """Load all Agglomerative Clustering artifacts exported by the notebook."""
    agg_dir = MODELS_DIR / "agg_clustering_artifacts"
    try:
        agg_scaler = joblib.load(agg_dir / "agg_scaler.pkl")
        agg_centroids = joblib.load(agg_dir / "agg_centroids.npy")
        agg_profile_mean = joblib.load(agg_dir / "agg_profile_mean.pkl")
        agg_profile_median = joblib.load(agg_dir / "agg_profile_median.pkl")
        agg_summary = joblib.load(agg_dir / "agg_cluster_summary.pkl")
        agg_eval_metrics = joblib.load(agg_dir / "agg_eval_metrics.pkl")
        agg_eval_by_k = joblib.load(agg_dir / "agg_eval_by_k.pkl")
        agg_config = joblib.load(agg_dir / "agg_config.pkl")
        return {
            "scaler": agg_scaler,
            "centroids": agg_centroids,
            "profile_mean": agg_profile_mean,
            "profile_median": agg_profile_median,
            "summary": agg_summary,
            "eval_metrics": agg_eval_metrics,
            "eval_by_k": agg_eval_by_k,
            "config": agg_config,
            "images_dir": agg_dir,
        }
    except (FileNotFoundError, Exception):
        return None


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
                <span style="color:#626478;">R&sup2; Score: <b>0.892</b> &bull; RMSE: <b>&pound;3,940</b></span>
            </p>
        </div>""", unsafe_allow_html=True)
        st.button("Explore Regression →", key="btn_reg", width='stretch',
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
                <span style="color:#626478;">Weighted F1: <b>0.883</b> &bull; ROC-AUC: <b>0.863</b></span>
            </p>
        </div>""", unsafe_allow_html=True)
        st.button("Explore Classification →", key="btn_clf", width='stretch',
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
                <span style="color:#626478;">Silhouette: <b>0.542</b> &bull; Calinski-H: <b>15,991</b></span>
            </p>
        </div>""", unsafe_allow_html=True)
        st.button("Explore Clustering →", key="btn_clust", width='stretch',
                  on_click=go_to, args=("Clustering",))

    st.markdown('<p class="section-header">Verified Model Performance Highlights</p>', unsafe_allow_html=True)
    home_metrics = [
        ("0.892", "Regression R²"),
        ("£3,940", "Regression RMSE"),
        ("89.0%", "Classifier Accuracy"),
        ("0.883", "Classifier F1"),
        ("0.863", "Classifier AUC"),
        ("0.542", "Cluster Silhouette (K=4)"),
    ]
    home_metric_cards = "".join(
        f'<div class="metric-card"><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>'
        for value, label in home_metrics
    )
    st.markdown(
        f'<div class="home-metrics">{home_metric_cards}</div>',
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
            submitted = st.form_submit_button("Predict Daily Sales Volume", width='stretch')

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
            width='stretch', hide_index=True,
        )

        # Regression Visualizations
        st.markdown('<p class="section-header">Model Diagnostics & Visualizations</p>', unsafe_allow_html=True)
        reg_viz_tab1, reg_viz_tab2, reg_viz_tab3, reg_viz_tab4 = st.tabs([
            "Actual vs Predicted", "Residuals", "Feature Importance", "Model Comparison"
        ])

        # Load test data for visualizations (mirror export_models.py logic)
        reg_df = pd.read_csv(ROOT / "data" / "raw" / "data_1.csv", encoding="ISO-8859-1")
        reg_df["Date"] = pd.to_datetime(reg_df["Date"], errors="coerce")
        reg_df["year"] = reg_df["Date"].dt.year
        reg_df["month"] = reg_df["Date"].dt.month
        reg_df["day_of_week"] = reg_df["Date"].dt.dayofweek
        reg_df.drop(columns=["Date"], inplace=True)
        reg_df["avg_items_per_order"] = reg_df["total_items_sold"] / reg_df["total_orders"]

        TARGET = "daily_sales_volume"
        X = reg_df.drop(columns=[TARGET])
        y_original = reg_df[TARGET].clip(lower=0)
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
        outlier_bounds_viz = {}
        for col in continuous_features:
            if col not in X_train.columns:
                continue
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lb = Q1 - 1.5 * IQR
            ub = Q3 + 1.5 * IQR
            outlier_bounds_viz[col] = (lb, ub)
            X_train[col] = X_train[col].clip(lb, ub)
            X_test[col] = X_test[col].clip(lb, ub)

        # Group rare countries
        country_counts = X_train["Country"].value_counts()
        rare = country_counts[country_counts < 5].index
        X_train["Country"] = X_train["Country"].replace(rare, "Other")
        X_test["Country"] = X_test["Country"].replace(rare, "Other")

        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        preprocessor_viz = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
            ]
        )

        X_train_proc = preprocessor_viz.fit_transform(X_train)
        X_test_proc = preprocessor_viz.transform(X_test)

        # Get predictions
        y_pred_log = model.predict(X_test_proc)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 0)
        residuals = y_test_orig - y_pred

        with reg_viz_tab1:
            # Actual vs Predicted
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(y_test_orig, y_pred, alpha=0.5, color="#8B83C9", edgecolors="none", s=30)
            max_val = max(y_test_orig.max(), y_pred.max())
            ax.plot([0, max_val], [0, max_val], 'k--', lw=2, label="Perfect Prediction")
            ax.set_xlabel("Actual Daily Sales Volume (£)")
            ax.set_ylabel("Predicted Daily Sales Volume (£)")
            ax.set_title("Actual vs Predicted — Tuned Random Forest")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        with reg_viz_tab2:
            # Residual plots
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            
            # Residuals vs Predicted
            axes[0].scatter(y_pred, residuals, alpha=0.5, color="#A8C9B5", edgecolors="none", s=30)
            axes[0].axhline(0, color='k', linestyle='--', lw=1.5)
            axes[0].set_xlabel("Predicted Daily Sales Volume (£)")
            axes[0].set_ylabel("Residual (£)")
            axes[0].set_title("Residuals vs Predicted")
            axes[0].grid(True, alpha=0.3)
            
            # Residual Distribution
            axes[1].hist(residuals, bins=40, color="#E8B9A8", edgecolor="#E6E4EF", alpha=0.8, density=True)
            axes[1].axvline(0, color='k', linestyle='--', lw=1.5)
            axes[1].set_xlabel("Residual (£)")
            axes[1].set_ylabel("Density")
            axes[1].set_title("Residual Distribution")
            axes[1].grid(True, alpha=0.3)
            
            fig.tight_layout()
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        with reg_viz_tab3:
            # Feature Importance (Random Forest)
            if hasattr(model, "feature_importances_"):
                feature_names = preprocessor.get_feature_names_out()
                importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(8, 5))
                importances[::-1].plot(kind="barh", color="#8B83C9", ax=ax)
                ax.set_title("Feature Importance — Tuned Random Forest (MDI)")
                ax.set_xlabel("Mean Impurity Reduction")
                fig.tight_layout()
                st.pyplot(fig, width='stretch')
                plt.close(fig)
                
                st.caption("Mean Decrease in Impurity (MDI) importance. Higher values indicate features that contribute more to reducing variance across trees.")
            else:
                st.info("Feature importance not available for this model type.")

        with reg_viz_tab4:
            # Model comparison bar chart
            fig, ax = plt.subplots(figsize=(10, 5.5))
            colors = ["#8B83C9" if "Selected" in m else "#A9B9E8" for m in reg_table["Model"]]
            bars = ax.barh(reg_table["Model"], reg_table["R2"], color=colors, height=0.6)
            ax.set_xlabel("R² Score (Higher is Better)")
            ax.set_title("Regression Models — R² Score Comparison")
            ax.set_xlim(0.75, 0.92)
            ax.axvline(0.85, linestyle="--", color="#626478", linewidth=1, label="0.85 Benchmark")
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            ax.legend(loc="lower right")
            # Add value labels
            for bar, val in zip(bars, reg_table["R2"]):
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.4f}", 
                        va='center', ha='left', fontsize=9, fontweight='bold')
            fig.tight_layout()
            st.pyplot(fig, width='stretch')
            plt.close(fig)


# ════════════════════ CLASSIFICATION ═════════════════════════════════════════

elif page == "Classification":
    render_page_header(
        "Online Purchase Intent Prediction",
        "Predicts whether a browsing session concludes with a transaction (<b>Revenue = True</b>). "
        "Ten classifiers trained with zero-leakage stratified scaling — select a model to explore."
    )

    pre_data, all_models = load_all_classification_models()
    if not all_models:
        artifact_missing_warning("Classification")
    else:
        scaler = pre_data["scaler"]
        encoder = pre_data["encoder"]

        # Model selector
        model_display_names = {
            "logistic": "Logistic Regression",
            "knn": "K-Nearest Neighbors (k=13, Manhattan)",
            "nb": "Gaussian Naive Bayes",
            "dt": "Decision Tree (depth=5, leaf=10)",
            "svm": "SVM RBF (C=10)",
            "rf": "Random Forest (200 trees, depth=20)",
            "ada": "AdaBoost (50 stumps, lr=0.5)",
            "gb": "Gradient Boosting (100 rounds, lr=0.05, depth=3) [Selected]",
            "bag": "Bagging (100 unpruned trees)",
            "mlp": "MLP (100 ReLU, α=0.001)",
        }
        available_models = {k: v for k, v in model_display_names.items() if k in all_models}
        
        selected_key = st.selectbox(
            "Select Model",
            options=list(available_models.keys()),
            format_func=lambda k: available_models[k],
            index=list(available_models.keys()).index("gb") if "gb" in available_models else 0,
            key="clf_model_selector",
        )
        model = all_models[selected_key]

        # Model metrics from notebook evaluation
        model_metrics = {
            "logistic": {"Accuracy": 0.8812, "F1": 0.8630, "ROC-AUC": 0.8875, "Precision": 0.7432, "Recall": 0.3560, "F1_buy": 0.4814},
            "knn": {"Accuracy": 0.8771, "F1": 0.8575, "ROC-AUC": 0.8141, "Precision": 0.7426, "Recall": 0.3351, "F1_buy": 0.3900},
            "nb": {"Accuracy": 0.7936, "F1": 0.8110, "ROC-AUC": 0.8026, "Precision": 0.3012, "Recall": 0.7696, "F1_buy": 0.4330},
            "dt": {"Accuracy": 0.8974, "F1": 0.8931, "ROC-AUC": 0.9177, "Precision": 0.7061, "Recall": 0.5785, "F1_buy": 0.6360},
            "svm": {"Accuracy": 0.8901, "F1": 0.8832, "ROC-AUC": 0.8627, "Precision": 0.6960, "Recall": 0.5000, "F1_buy": 0.5823},
            "rf": {"Accuracy": 0.9015, "F1": 0.8953, "ROC-AUC": 0.9136, "Precision": 0.7456, "Recall": 0.5524, "F1_buy": 0.6346},
            "ada": {"Accuracy": 0.8913, "F1": 0.8894, "ROC-AUC": 0.9130, "Precision": 0.6629, "Recall": 0.6073, "F1_buy": 0.6339},
            "gb": {"Accuracy": 0.9015, "F1": 0.8971, "ROC-AUC": 0.9300, "Precision": 0.7249, "Recall": 0.5864, "F1_buy": 0.6483},
            "bag": {"Accuracy": 0.8986, "F1": 0.8949, "ROC-AUC": 0.9177, "Precision": 0.7063, "Recall": 0.5916, "F1_buy": 0.6439},
            "mlp": {"Accuracy": 0.8889, "F1": 0.8799, "ROC-AUC": 0.9037, "Precision": 0.7077, "Recall": 0.4817, "F1_buy": 0.5732},
        }

        m = model_metrics[selected_key]
        st.markdown(f'<p class="section-header">{available_models[selected_key]} — Test Set Performance</p>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5, m6 = st.columns(6, gap="medium")
        for col, val, lbl in [
            (m1, f"{m['Accuracy']*100:.1f}%", "Accuracy"),
            (m2, f"{m['F1']:.4f}", "Weighted F1"),
            (m3, f"{m['ROC-AUC']:.4f}", "ROC-AUC"),
            (m4, f"{m['Precision']*100:.1f}%", "Pos. Precision"),
            (m5, f"{m['Recall']*100:.1f}%", "Pos. Recall"),
            (m6, f"{m['F1_buy']:.4f}", "Pos. F1"),
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="value">{val}</div><div class="label">{lbl}</div></div>',
                unsafe_allow_html=True)

        # Tabs for visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Confusion Matrix", "ROC / PR Curves", "Feature Importance", "Predict"])

        with viz_tab1:
            # We need test data for confusion matrix - load it
            from nb_utils import load_and_preprocess_data
            data = load_and_preprocess_data(test_size=0.20, random_state=42)
            X_test = data["X_test"]
            y_test = data["y_test"]
            
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(f"{available_models[selected_key]} — Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticklabels(["No Purchase", "Purchase"])
            ax.set_yticklabels(["No Purchase", "Purchase"])
            st.pyplot(fig, width='stretch')
            plt.close(fig)
            
            tn, fp, fn, tp = cm.ravel()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("True Negatives", f"{tn:,}")
            c2.metric("False Positives", fp)
            c3.metric("False Negatives", fn)
            c4.metric("True Positives", tp)

        with viz_tab2:
            # ROC Curve
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            else:
                y_score = model.predict(X_test).astype(float)

            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = roc_auc_score(y_test, y_score)
            precision, recall, _ = precision_recall_curve(y_test, y_score)
            pr_auc = average_precision_score(y_test, y_score)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}", color="#8B83C9", lw=2)
            axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="No Skill (0.50)")
            axes[0].set_xlabel("False Positive Rate")
            axes[0].set_ylabel("True Positive Rate (Recall)")
            axes[0].set_title("ROC Curve")
            axes[0].legend(loc="lower right")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}", color="#A8C9B5", lw=2)
            axes[1].axhline(y_test.mean(), color="black", linestyle="--", lw=1, label=f"Baseline ({y_test.mean():.3f})")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].set_title("Precision-Recall Curve")
            axes[1].legend(loc="lower left")
            axes[1].grid(True, alpha=0.3)

            fig.tight_layout()
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        with viz_tab3:
            # Feature importance / coefficients
            if hasattr(model, "feature_importances_"):
                importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(8, 5))
                importances[::-1].plot(kind="barh", color="#8B83C9", ax=ax)
                ax.set_title("Feature Importance (MDI)")
                ax.set_xlabel("Mean Impurity Reduction")
                fig.tight_layout()
                st.pyplot(fig, width='stretch')
                plt.close(fig)
            elif hasattr(model, "coef_"):
                coef = pd.Series(model.coef_[0], index=X_test.columns).sort_values(key=abs, ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ["#28573d" if c > 0 else "#8c4633" for c in coef[::-1]]
                coef[::-1].plot(kind="barh", color=colors, ax=ax)
                ax.axvline(0, color="black", lw=0.5)
                ax.set_title("Logistic Regression Coefficients")
                ax.set_xlabel("Coefficient Value")
                fig.tight_layout()
                st.pyplot(fig, width='stretch')
                plt.close(fig)
            else:
                st.info("Feature importance not available for this model type.")

        with viz_tab4:
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
                    c_m1, c_m2 = st.columns(2)
                    month = c_m1.selectbox("Session Month", month_options)
                    visitor_type = c_m2.selectbox("Visitor Type", visitor_options)
                    weekend = st.selectbox("Weekend Session?", ["False", "True"])

                    t1, t2 = st.columns(2)
                    os = t1.number_input("Operating System (1-8)", 1, 8, 2)
                    browser = t2.number_input("Browser (1-13)", 1, 13, 2)

                    t3, t4 = st.columns(2)
                    region = t3.number_input("Region (1-9)", 1, 9, 1)
                    traffic = t4.number_input("Traffic Type (1-20)", 1, 20, 2)

                st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Predict Purchase Intent", width='stretch')

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

        # All models comparison table
        st.markdown('<p class="section-header">All 10 Classifiers Comparison (Test Set)</p>', unsafe_allow_html=True)
        clf_table = pd.DataFrame([
            {"Algorithm": "Gradient Boosting [Selected]",          "Accuracy": 0.9015, "F1 (Weighted)": 0.8971, "ROC-AUC": 0.9300, "Pos. Recall": 0.586, "Pos. F1": 0.6483},
            {"Algorithm": "Random Forest",                 "Accuracy": 0.9015, "F1 (Weighted)": 0.8953, "ROC-AUC": 0.9136, "Pos. Recall": 0.552, "Pos. F1": 0.6346},
            {"Algorithm": "Bagging",                       "Accuracy": 0.8986, "F1 (Weighted)": 0.8949, "ROC-AUC": 0.9177, "Pos. Recall": 0.592, "Pos. F1": 0.6439},
            {"Algorithm": "Decision Tree (depth=5)",       "Accuracy": 0.8974, "F1 (Weighted)": 0.8931, "ROC-AUC": 0.9177, "Pos. Recall": 0.579, "Pos. F1": 0.6360},
            {"Algorithm": "AdaBoost",                      "Accuracy": 0.8913, "F1 (Weighted)": 0.8894, "ROC-AUC": 0.9130, "Pos. Recall": 0.607, "Pos. F1": 0.6339},
            {"Algorithm": "SVM RBF (C=10)",                "Accuracy": 0.8901, "F1 (Weighted)": 0.8832, "ROC-AUC": 0.8627, "Pos. Recall": 0.500, "Pos. F1": 0.5823},
            {"Algorithm": "Logistic Regression",           "Accuracy": 0.8812, "F1 (Weighted)": 0.8630, "ROC-AUC": 0.8875, "Pos. Recall": 0.356, "Pos. F1": 0.4814},
            {"Algorithm": "MLP",                           "Accuracy": 0.8889, "F1 (Weighted)": 0.8799, "ROC-AUC": 0.9037, "Pos. Recall": 0.482, "Pos. F1": 0.5732},
            {"Algorithm": "KNN (k=13, manhattan)",         "Accuracy": 0.8719, "F1 (Weighted)": 0.8450, "ROC-AUC": 0.8141, "Pos. Recall": 0.264, "Pos. F1": 0.3900},
            {"Algorithm": "Gaussian Naive Bayes",          "Accuracy": 0.7936, "F1 (Weighted)": 0.8110, "ROC-AUC": 0.8026, "Pos. Recall": 0.770, "Pos. F1": 0.4330},
            {"Algorithm": "Dummy Baseline",                "Accuracy": 0.8451, "F1 (Weighted)": 0.7741, "ROC-AUC": 0.5000, "Pos. Recall": 0.000, "Pos. F1": 0.0000},
        ]).sort_values("F1 (Weighted)", ascending=False)
        st.dataframe(
            clf_table.style.format({
                "Accuracy": "{:.1%}", "F1 (Weighted)": "{:.4f}",
                "ROC-AUC": "{:.4f}", "Pos. Recall": "{:.1%}", "Pos. F1": "{:.4f}",
            }),
            width='stretch', hide_index=True,
        )


# ════════════════════ CLUSTERING ══════════════════════════════════════════════

elif page == "Clustering":
    render_page_header(
        "E-Commerce Session Clustering",
        "Segmenting e-commerce user sessions into distinct behavioral clusters "
        "using <b>MiniBatch K-Means</b> or <b>Agglomerative Hierarchical Clustering</b>."
    )

    clustering_algo = st.radio(
        "Select Clustering Algorithm",
        ["MiniBatch K-Means", "Agglomerative Hierarchical Clustering"],
        horizontal=True,
        key="clustering_algo_selector",
    )
    st.divider()

    # ──────────────────── MiniBatch K-Means Dashboard ────────────────────────
    if clustering_algo == "MiniBatch K-Means":
        st.markdown(
            '<p style="font-size:0.92rem;color:#626478;margin-bottom:1rem;">'
            'Segmenting <b>9.24 million</b> e-commerce user sessions into <b>4 distinct behavioral clusters</b> '
            'using <b>MiniBatch K-Means (K=4)</b> on session-level aggregated activity.</p>',
            unsafe_allow_html=True,
        )

        scaler, km_model, cluster_profile, cluster_summary = load_clustering_artifacts()

        def cluster_value(cluster_id, feature, fallback):
            if cluster_profile is not None:
                if cluster_id not in cluster_profile.index:
                    return 0.0
                return float(cluster_profile.loc[cluster_id, feature])
            return fallback

        def cluster_percentage(cluster_id, fallback):
            if cluster_summary is not None:
                if cluster_id not in cluster_summary.index:
                    return 0.0
                return float(cluster_summary.loc[cluster_id, "Percentage (%)"])
            return fallback

        st.markdown('<p class="section-header">Identified E-Commerce Session Segments</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            st.markdown(f"""<div class="cluster-card" style="border-top:3.5px solid #8B83C9;">
                <div>
                    <span class="badge badge-accent">CLUSTER 0 &bull; {cluster_percentage(0, 15.49):.1f}%</span>
                    <h4>High-Value Browsers</h4>
                    <p>High viewed item prices (&pound;{cluster_value(0, "avg_price_viewed", 963.20):,.2f} avg), {cluster_value(0, "total_views", 3.62):.1f} views, {cluster_value(0, "cart_additions", 0.0):.2f} carts. Luxury / high-ticket interest without immediate commitment.</p>
                </div>
                <p style="margin-top:0.6rem;padding-top:0.4rem;border-top:1px solid #E6E4EF;font-size:0.82rem;color:#626478;">
                    <b>Focus:</b> Premium Window Shopping
                </p>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""<div class="cluster-card" style="border-top:3.5px solid #A9B9E8;">
                <div>
                    <span class="badge badge-regression">CLUSTER 1 &bull; {cluster_percentage(1, 73.24):.1f}%</span>
                    <h4>Casual Browsers</h4>
                    <p>Low price viewed (&pound;{cluster_value(1, "avg_price_viewed", 180.54):,.2f} avg), ~{cluster_value(1, "total_views", 3.14):.1f} page views, {cluster_value(1, "cart_additions", 0.0):.2f} carts. Represents standard fast window shoppers.</p>
                </div>
                <p style="margin-top:0.6rem;padding-top:0.4rem;border-top:1px solid #E6E4EF;font-size:0.82rem;color:#626478;">
                    <b>Focus:</b> Quick Casual Visits
                </p>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""<div class="cluster-card" style="border-top:3.5px solid #A8C9B5;">
                <div>
                    <span class="badge badge-classification">CLUSTER 2 &bull; {cluster_percentage(2, 5.26):.1f}%</span>
                    <h4>Catalog Researchers</h4>
                    <p>Extensive depth ({cluster_value(2, "unique_products_viewed", 15.59):.2f} unique SKUs, {cluster_value(2, "total_views", 23.42):.2f} views, {cluster_value(2, "session_duration_minutes", 37.43):.2f} min duration). Suggests comparative product research.</p>
                </div>
                <p style="margin-top:0.6rem;padding-top:0.4rem;border-top:1px solid #E6E4EF;font-size:0.82rem;color:#626478;">
                    <b>Focus:</b> Deep In-Catalog Evaluation
                </p>
            </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""<div class="cluster-card" style="border-top:3.5px solid #E8B9A8;">
                <div>
                    <span class="badge badge-clustering">CLUSTER 3 &bull; {cluster_percentage(3, 6.01):.1f}%</span>
                    <h4>Carting Shoppers</h4>
                    <p>Active cart additions ({cluster_value(3, "cart_additions", 1.58):.2f} avg, {cluster_value(3, "unique_products_carted", 1.10):.2f} unique products). Indicates comparatively stronger purchase-intent signals, not a guaranteed conversion.</p>
                </div>
                <p style="margin-top:0.6rem;padding-top:0.4rem;border-top:1px solid #E6E4EF;font-size:0.82rem;color:#626478;">
                    <b>Focus:</b> Direct Pre-Purchase Intent
                </p>
            </div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-header">MiniBatch K-Means (K=4) — Clustering Validation</p>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3, gap="medium")
        for col, val, lbl in [
            (m1, "0.542", "Silhouette Score (higher = better)"),
            (m2, "0.914", "Davies-Bouldin Index (lower = better)"),
            (m3, "15,991", "Calinski-Harabasz Index (higher = better)"),
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="value">{val}</div><div class="label">{lbl}</div></div>',
                unsafe_allow_html=True)
        st.caption("Evaluated on a 50,000-session stratified sample across K=4 clusters.")

        st.markdown('<p class="section-header">Cluster Distribution</p>', unsafe_allow_html=True)
        col_dist, col_chart = st.columns([1, 2], gap="large")
        with col_dist:
            if cluster_summary is not None:
                dist_df = cluster_summary.rename_axis("Cluster").reset_index()
                dist_df["Segment"] = dist_df["Cluster"].map({
                    0: "High-Value Browsers",
                    1: "Casual Browsers",
                    2: "Catalog Researchers",
                    3: "Carting Shoppers",
                })
                dist_df = dist_df[["Cluster", "Segment", "Sessions", "Percentage (%)"]]
            else:
                dist_df = pd.DataFrame({
                    "Cluster": [0, 1, 2, 3],
                    "Segment": ["High-Value Browsers", "Casual Browsers", "Catalog Researchers", "Carting Shoppers"],
                    "Sessions": [1431933, 6770409, 486228, 555851],
                    "Percentage (%)": [15.49, 73.24, 5.26, 6.01]
                })
            st.dataframe(
                dist_df.style.format({"Sessions": "{:,.0f}", "Percentage (%)": "{:.2f}%"}),
                width='stretch', hide_index=True
            )

        with col_chart:
            fig_km_dist, ax_km_dist = plt.subplots(figsize=(7, 3.5))
            cluster_colors = ["#8B83C9", "#A9B9E8", "#A8C9B5", "#E8B9A8"]
            km_bars = ax_km_dist.bar(
                [f"Cluster {c}" for c in dist_df["Cluster"]],
                dist_df["Sessions"],
                color=cluster_colors[:len(dist_df)],
                edgecolor="#E6E4EF",
                linewidth=0.8,
            )
            for bar, pct in zip(km_bars, dist_df["Percentage (%)"]):
                ax_km_dist.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{pct:.1f}%",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#303247",
                )
            ax_km_dist.set_xlabel("Cluster")
            ax_km_dist.set_ylabel("Number of Sessions")
            ax_km_dist.set_title("Session Distribution by MiniBatch K-Means Cluster")
            ax_km_dist.grid(axis="y", linestyle="--", alpha=0.7)
            ax_km_dist.set_axisbelow(True)
            fig_km_dist.tight_layout()
            st.pyplot(fig_km_dist, width='stretch')
            plt.close(fig_km_dist)

        st.markdown('<p class="section-header">Cluster Feature Averages Profile</p>', unsafe_allow_html=True)
        if cluster_profile is not None:
            prof_df = cluster_profile.rename_axis("Cluster").reset_index().rename(columns={
                "unique_products_viewed": "Products",
                "unique_categories_viewed": "Categories",
                "total_views": "Views",
                "cart_additions": "Carts",
                "unique_products_carted": "Products Carted",
                "avg_price_viewed": "Avg Price (£)",
                "max_price_viewed": "Max Price (£)",
                "session_duration_minutes": "Duration (min)",
            })
        else:
            prof_df = pd.DataFrame([
                {"Cluster": 0, "Products": 2.70, "Categories": 1.22, "Views": 3.62, "Carts": 0.00, "Products Carted": 0.00, "Avg Price (£)": 963.20, "Max Price (£)": 1221.72, "Duration (min)": 16.67},
                {"Cluster": 1, "Products": 2.20, "Categories": 1.20, "Views": 3.14, "Carts": 0.00, "Products Carted": 0.00, "Avg Price (£)": 180.54, "Max Price (£)": 217.25, "Duration (min)": 16.54},
                {"Cluster": 2, "Products": 15.59, "Categories": 3.93, "Views": 23.42, "Carts": 0.03, "Products Carted": 0.03, "Avg Price (£)": 232.62, "Max Price (£)": 683.06, "Duration (min)": 37.43},
                {"Cluster": 3, "Products": 2.65, "Categories": 1.23, "Views": 5.31, "Carts": 1.58, "Products Carted": 1.10, "Avg Price (£)": 332.88, "Max Price (£)": 414.66, "Duration (min)": 11.62},
            ])
        st.dataframe(
            prof_df.style.format({
                "Products": "{:.2f}", "Categories": "{:.2f}", "Views": "{:.2f}",
                "Carts": "{:.2f}", "Products Carted": "{:.2f}",
                "Avg Price (£)": "{:,.2f}", "Max Price (£)": "{:,.2f}",
                "Duration (min)": "{:.2f}"
            }),
            width='stretch', hide_index=True
        )

        if km_model is not None:
            st.markdown('<p class="section-header">Assign a New Session to a Segment</p>', unsafe_allow_html=True)

            with st.form("cluster_form"):
                c1, c2 = st.columns(2, gap="large")
                with c1:
                    st.markdown("<p style='font-size:0.92rem;font-weight:600;color:#303247;margin:0 0 0.5rem;'>Browsing & Cart Activity</p>", unsafe_allow_html=True)
                    uniq_prod = st.number_input("Unique Products Viewed", 0, 500, 3)
                    uniq_cat = st.number_input("Unique Categories Viewed", 0, 100, 1)
                    total_views = st.number_input("Total Page Views", 0, 1000, 4)
                    cart_add = st.number_input("Cart Additions", 0, 200, 0)
                with c2:
                    st.markdown("<p style='font-size:0.92rem;font-weight:600;color:#303247;margin:0 0 0.5rem;'>Price Range & Duration</p>", unsafe_allow_html=True)
                    uniq_carted = st.number_input("Unique Products Carted", 0, 200, 0)
                    avg_price = st.number_input("Avg Price Viewed (£)", 0.0, 5000.0, 50.0, step=0.5)
                    max_price = st.number_input("Max Price Viewed (£)", 0.0, 10000.0, 100.0, step=0.5)
                    session_dur = st.number_input("Session Duration (minutes)", 0.0, 600.0, 5.0, step=0.5)

                st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Assign to Cluster", width='stretch')

            if submitted:
                row = pd.DataFrame([[
                    uniq_prod, uniq_cat, total_views, cart_add, uniq_carted,
                    avg_price, max_price, session_dur,
                ]], columns=[
                    "unique_products_viewed", "unique_categories_viewed", "total_views",
                    "cart_additions", "unique_products_carted", "avg_price_viewed",
                    "max_price_viewed", "session_duration_minutes",
                ])
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
            width='stretch', hide_index=True,
        )

        if km_model is None:
            st.markdown(
                '<div class="error-box" style="margin-top:1.5rem;padding:0.75rem 1rem;background-color:rgba(232,185,168,0.08);border:1px solid rgba(232,185,168,0.3);border-radius:8px;font-size:0.86rem;color:#626478;line-height:1.5;">'
                '<b>Note:</b> Cluster assignment feature requires model artifacts. '
                'To generate from the 5.3GB dataset, run: <code style="background-color:rgba(255,255,255,0.7);color:#8c4633;padding:0.1rem 0.35rem;border-radius:3px;border:1px solid rgba(232,185,168,0.3);font-size:0.8rem;">python export_models.py --task clustering</code>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ──────────────── Agglomerative Hierarchical Clustering Dashboard ────────
    else:
        agg_data = load_agg_clustering_artifacts()
        if agg_data is None:
            st.markdown(
                '<div class="pastel-notice"><b>Agglomerative Clustering artifacts not found.</b><br>'
                'Run the export cell at the end of <code>notebooks/agg_clustering.ipynb</code> to generate '
                'artifacts to <code>models/agg_clustering_artifacts/</code>.</div>',
                unsafe_allow_html=True,
            )
        else:
            agg_config = agg_data["config"]
            agg_eval = agg_data["eval_metrics"]
            agg_eval_k = agg_data["eval_by_k"]
            agg_summary = agg_data["summary"]
            agg_mean = agg_data["profile_mean"]
            agg_median = agg_data["profile_median"]
            agg_centroids = agg_data["centroids"]
            agg_scaler = agg_data["scaler"]
            agg_images = agg_data["images_dir"]
            agg_features = agg_config["cluster_features"]

            # ── A. Model Overview ──
            st.markdown('<p class="section-header">Model Overview</p>', unsafe_allow_html=True)
            ov1, ov2, ov3, ov4 = st.columns(4, gap="medium")
            ov1.markdown(
                '<div class="metric-card"><div class="value">Agglomerative</div>'
                '<div class="label">Algorithm</div></div>', unsafe_allow_html=True)
            ov2.markdown(
                f'<div class="metric-card"><div class="value">Ward</div>'
                f'<div class="label">Linkage Method</div></div>', unsafe_allow_html=True)
            ov3.markdown(
                f'<div class="metric-card"><div class="value">{agg_config["n_clusters"]}</div>'
                f'<div class="label">Number of Clusters</div></div>', unsafe_allow_html=True)
            ov4.markdown(
                f'<div class="metric-card"><div class="value">{agg_config["hierarchical_sample_size"]:,}</div>'
                f'<div class="label">Hierarchical Fit Sample</div></div>', unsafe_allow_html=True)

            st.markdown(
                '<div class="pastel-notice">'
                '<b>Scalability note:</b> Ward-linkage Agglomerative Clustering was fitted on a reproducible '
                f'sample of <b>{agg_config["hierarchical_sample_size"]:,}</b> sessions (out of '
                f'<b>{agg_config["n_eligible_sessions"]:,}</b> eligible). '
                'Remaining sessions were assigned to their <b>nearest cluster centroid</b> in the same '
                'scaled feature space. These centroid assignments are <b>approximate</b> — '
                'AgglomerativeClustering does not support a native <code>predict()</code> method.'
                '</div>',
                unsafe_allow_html=True,
            )

            # ── B. Evaluation Metrics ──
            st.markdown('<p class="section-header">Evaluation Metrics (Fitted Sample)</p>', unsafe_allow_html=True)
            em1, em2, em3 = st.columns(3, gap="medium")
            sil_val = agg_eval.get("Silhouette Score", 0)
            db_val = agg_eval.get("Davies-Bouldin Index", 0)
            ch_val = agg_eval.get("Calinski-Harabasz Index", 0)
            em1.markdown(
                f'<div class="metric-card"><div class="value">{sil_val:.4f}</div>'
                f'<div class="label">Silhouette Score (higher = better)</div></div>',
                unsafe_allow_html=True)
            em2.markdown(
                f'<div class="metric-card"><div class="value">{db_val:.4f}</div>'
                f'<div class="label">Davies-Bouldin Index (lower = better)</div></div>',
                unsafe_allow_html=True)
            em3.markdown(
                f'<div class="metric-card"><div class="value">{ch_val:,.2f}</div>'
                f'<div class="label">Calinski-Harabasz Index (higher = better)</div></div>',
                unsafe_allow_html=True)
            st.caption(
                f"Metrics evaluated on the {agg_config['hierarchical_sample_size']:,}-session "
                f"fitted sample with exact hierarchical labels."
            )

            # ── C. Cluster Distribution ──
            st.markdown('<p class="section-header">Cluster Distribution</p>', unsafe_allow_html=True)
            col_dist, col_chart = st.columns([1, 2], gap="large")
            with col_dist:
                agg_dist_df = agg_summary.rename_axis("Cluster").reset_index()
                st.dataframe(
                    agg_dist_df.style.format({"Sessions": "{:,.0f}", "Percentage": "{:.2f}%"}),
                    width='stretch', hide_index=True,
                )

            with col_chart:
                configure_matplotlib_theme()
                fig_dist, ax_dist = plt.subplots(figsize=(7, 3.5))
                cluster_colors = ["#8B83C9", "#A9B9E8", "#A8C9B5", "#E8B9A8"]
                bars = ax_dist.bar(
                    agg_dist_df["Cluster"].astype(str),
                    agg_dist_df["Sessions"],
                    color=cluster_colors[:len(agg_dist_df)],
                    edgecolor="#E6E4EF",
                    linewidth=0.8,
                )
                for bar, pct in zip(bars, agg_dist_df["Percentage"]):
                    ax_dist.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{pct:.1f}%",
                        ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#303247",
                    )
                ax_dist.set_xlabel("Cluster")
                ax_dist.set_ylabel("Number of Sessions")
                ax_dist.set_title("Session Distribution by Agglomerative Cluster")
                ax_dist.grid(axis="y", linestyle="--", alpha=0.7)
                ax_dist.set_axisbelow(True)
                fig_dist.tight_layout()
                st.pyplot(fig_dist, width='stretch')
                plt.close(fig_dist)

            # ── D. Cluster Profiles ──
            st.markdown('<p class="section-header">Cluster Profiles</p>', unsafe_allow_html=True)
            agg_col_rename = {
                "unique_products_viewed": "Products Viewed",
                "unique_categories_viewed": "Categories Viewed",
                "total_views": "Total Views",
                "cart_additions": "Cart Additions",
                "unique_products_carted": "Products Carted",
                "avg_price_viewed": "Avg Price Viewed (£)",
                "session_duration_minutes": "Duration (min)",
            }
            profile_tab_mean, profile_tab_median = st.tabs(["Mean Profile", "Median Profile"])
            with profile_tab_mean:
                mean_df = agg_mean.rename_axis("Cluster").reset_index().rename(columns=agg_col_rename)
                fmt_dict = {v: "{:.2f}" for v in agg_col_rename.values()}
                fmt_dict["Avg Price Viewed (£)"] = "{:,.2f}"
                fmt_dict["Duration (min)"] = "{:,.2f}"
                st.dataframe(
                    mean_df.style.format(fmt_dict),
                    width='stretch', hide_index=True,
                )
            with profile_tab_median:
                median_df = agg_median.rename_axis("Cluster").reset_index().rename(columns=agg_col_rename)
                st.dataframe(
                    median_df.style.format(fmt_dict),
                    width='stretch', hide_index=True,
                )

            # ── E. Visualizations ──
            st.markdown('<p class="section-header">Visualizations</p>', unsafe_allow_html=True)
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Dendrogram", "PCA", "t-SNE"])

            with viz_tab1:
                dendro_path = agg_images / "dendrogram.png"
                if dendro_path.exists():
                    st.image(str(dendro_path), caption="Ward-Linkage Dendrogram (representative session sample)", width="stretch")
                else:
                    st.warning("Dendrogram image not found. Re-run the export cell in `agg_clustering.ipynb`.")

            with viz_tab2:
                pca_path = agg_images / "pca_clusters.png"
                if pca_path.exists():
                    st.image(str(pca_path), caption="PCA 2D Visualization of Agglomerative Clusters", width="stretch")
                else:
                    st.warning("PCA image not found. Re-run the export cell in `agg_clustering.ipynb`.")

            with viz_tab3:
                tsne_path = agg_images / "tsne_clusters.png"
                if tsne_path.exists():
                    st.image(str(tsne_path), caption="t-SNE Visualization of Agglomerative Clusters", width="stretch")
                else:
                    st.warning("t-SNE image not found. Re-run the export cell in `agg_clustering.ipynb`.")

            # ── F. Cluster-Count Evaluation ──
            st.markdown('<p class="section-header">Cluster Count Evaluation (K = 2–6)</p>', unsafe_allow_html=True)
            st.caption(
                f"Evaluated on a reproducible {agg_config.get('k_selection_sample_size', 2000):,}-session sample "
                f"with Ward-linkage Agglomerative Clustering fitted independently for each candidate K."
            )
            eval_k_df = agg_eval_k.reset_index()
            eval_k_display = eval_k_df.copy()
            eval_k_display["Selection"] = eval_k_display["K"].apply(
                lambda k: f"K={k} (Selected)" if k == agg_config["n_clusters"] else ""
            )
            fmt_eval = {
                "Silhouette Score": "{:.4f}",
                "Davies-Bouldin Index": "{:.4f}",
                "Calinski-Harabasz Index": "{:,.2f}",
                "Smallest Cluster (%)": "{:.1f}%",
                "Largest Cluster (%)": "{:.1f}%",
            }
            st.dataframe(
                eval_k_display.style.format(fmt_eval),
                width='stretch', hide_index=True,
            )

            # ── G. New-Session Assignment (Approximate) ──
            st.markdown('<p class="section-header">Estimate Cluster for a New Session (Approximate)</p>', unsafe_allow_html=True)
            st.markdown(
                '<div class="pastel-notice">'
                'This uses <b>nearest-centroid assignment</b> in the same scaled feature space '
                'as the notebook. The result is an <b>approximation</b> — it does not replicate '
                'the full hierarchical linkage.'
                '</div>',
                unsafe_allow_html=True,
            )

            with st.form("agg_cluster_form"):
                ac1, ac2 = st.columns(2, gap="large")
                with ac1:
                    st.markdown("<p style='font-size:0.92rem;font-weight:600;color:#303247;margin:0 0 0.5rem;'>Browsing & Cart Activity</p>", unsafe_allow_html=True)
                    agg_uniq_prod = st.number_input("Unique Products Viewed", 0, 500, 3, key="agg_uniq_prod")
                    agg_uniq_cat = st.number_input("Unique Categories Viewed", 0, 100, 1, key="agg_uniq_cat")
                    agg_total_views = st.number_input("Total Page Views", 0, 1000, 4, key="agg_total_views")
                    agg_cart_add = st.number_input("Cart Additions", 0, 200, 0, key="agg_cart_add")
                with ac2:
                    st.markdown("<p style='font-size:0.92rem;font-weight:600;color:#303247;margin:0 0 0.5rem;'>Valuation & Duration (7 Features)</p>", unsafe_allow_html=True)
                    agg_uniq_carted = st.number_input("Unique Products Carted", 0, 200, 0, key="agg_uniq_carted")
                    agg_avg_price = st.number_input("Avg Price Viewed (£)", 0.0, 5000.0, 50.0, step=0.5, key="agg_avg_price")
                    agg_session_dur = st.number_input("Session Duration (minutes)", 0.0, 100000.0, 5.0, step=0.5, key="agg_session_dur")

                st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
                agg_submitted = st.form_submit_button("Estimate Cluster (Approximate)", width='stretch')

            if agg_submitted:
                agg_row = pd.DataFrame([[
                    agg_uniq_prod, agg_uniq_cat, agg_total_views, agg_cart_add,
                    agg_uniq_carted, agg_avg_price, agg_session_dur,
                ]], columns=agg_features)
                try:
                    agg_row_scaled = agg_scaler.transform(agg_row)
                    # Nearest centroid assignment
                    distances = np.linalg.norm(agg_centroids - agg_row_scaled, axis=1)
                    agg_cluster_id = int(np.argmin(distances))
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="res-label">Approximate Cluster Assignment</div>'
                        f'<div class="res-value">Cluster {agg_cluster_id}</div>'
                        f'<div class="res-desc">Assigned by nearest-centroid in scaled feature space. '
                        f'This is an approximation; AgglomerativeClustering does not support native predict().</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Assignment failed: {e}")


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
        st.markdown('<p class="section-header" style="margin-top:0.5rem">Regression Models (Target: Daily Sales Volume &pound;)</p>', unsafe_allow_html=True)
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
            width='stretch', hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(9, 4.8))
        bar_colors = ["#8B83C9" if "Selected" in m else "#A9B9E8" for m in reg_df["Model"]]
        ax.barh(reg_df["Model"], reg_df["R2"], color=bar_colors, height=0.62)
        ax.set_xlabel("R² Score (Higher is Better)")
        ax.set_title("Regression Models — R² Score Comparison")
        ax.set_xlim(0.74, 0.92)
        ax.axvline(0.85, linestyle="--", color="#626478", linewidth=0.9, label="0.85 Benchmark")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    with tab2:
        st.markdown('<p class="section-header" style="margin-top:0.5rem">Classification Models (Target: Purchase Intent / Revenue)</p>', unsafe_allow_html=True)
        st.caption("Evaluated on 2,466 test sessions (20% stratified split). Baseline dummy accuracy is 84.51% with zero buyer capture. All 10 models trained on identical split with zero leakage.")
        
        clf_df = pd.DataFrame([
            {"Algorithm": "Gradient Boosting [Selected]",          "Accuracy": 0.9015, "F1 (Weighted)": 0.8971, "ROC-AUC": 0.9300, "PR-AUC": 0.7458, "Pos. Recall": 0.586, "Pos. F1": 0.6483},
            {"Algorithm": "Random Forest",                 "Accuracy": 0.9015, "F1 (Weighted)": 0.8953, "ROC-AUC": 0.9136, "PR-AUC": 0.7262, "Pos. Recall": 0.552, "Pos. F1": 0.6346},
            {"Algorithm": "Bagging",                       "Accuracy": 0.8986, "F1 (Weighted)": 0.8949, "ROC-AUC": 0.9177, "PR-AUC": 0.7140, "Pos. Recall": 0.592, "Pos. F1": 0.6439},
            {"Algorithm": "Decision Tree (depth=5)",       "Accuracy": 0.8974, "F1 (Weighted)": 0.8931, "ROC-AUC": 0.9177, "PR-AUC": 0.6882, "Pos. Recall": 0.579, "Pos. F1": 0.6360},
            {"Algorithm": "AdaBoost",                      "Accuracy": 0.8913, "F1 (Weighted)": 0.8894, "ROC-AUC": 0.9130, "PR-AUC": 0.6311, "Pos. Recall": 0.607, "Pos. F1": 0.6339},
            {"Algorithm": "SVM RBF (C=10)",                "Accuracy": 0.8901, "F1 (Weighted)": 0.8832, "ROC-AUC": 0.8627, "PR-AUC": 0.6646, "Pos. Recall": 0.500, "Pos. F1": 0.5823},
            {"Algorithm": "Logistic Regression",           "Accuracy": 0.8812, "F1 (Weighted)": 0.8630, "ROC-AUC": 0.8875, "PR-AUC": 0.6237, "Pos. Recall": 0.356, "Pos. F1": 0.4814},
            {"Algorithm": "MLP",                           "Accuracy": 0.8889, "F1 (Weighted)": 0.8799, "ROC-AUC": 0.9037, "PR-AUC": 0.6704, "Pos. Recall": 0.482, "Pos. F1": 0.5732},
            {"Algorithm": "KNN (k=13, manhattan)",         "Accuracy": 0.8719, "F1 (Weighted)": 0.8450, "ROC-AUC": 0.8141, "PR-AUC": 0.5318, "Pos. Recall": 0.264, "Pos. F1": 0.3900},
            {"Algorithm": "Gaussian Naive Bayes",          "Accuracy": 0.7936, "F1 (Weighted)": 0.8110, "ROC-AUC": 0.8026, "PR-AUC": 0.4475, "Pos. Recall": 0.770, "Pos. F1": 0.4330},
            {"Algorithm": "Dummy Baseline",                "Accuracy": 0.8451, "F1 (Weighted)": 0.7741, "ROC-AUC": 0.5000, "PR-AUC": 0.1549, "Pos. Recall": 0.000, "Pos. F1": 0.0000},
        ]).sort_values("F1 (Weighted)", ascending=False)

        st.dataframe(
            clf_df.style.format({
                "Accuracy": "{:.1%}", "F1 (Weighted)": "{:.4f}",
                "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}",
                "Pos. Recall": "{:.1%}", "Pos. F1": "{:.4f}",
            }),
            width='stretch', hide_index=True,
        )

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        no_dummy = clf_df[clf_df["Algorithm"] != "Dummy Baseline"].copy()
        colors = ["#8B83C9" if "[Selected]" in a else "#A9B9E8" for a in no_dummy["Algorithm"]]
        axes[0].barh(no_dummy["Algorithm"], no_dummy["F1 (Weighted)"], color=colors, height=0.55)
        axes[0].set_xlabel("Weighted F1 Score")
        axes[0].set_title("Weighted F1 Comparison")
        axes[0].grid(axis='x', linestyle='--', alpha=0.7)
        axes[0].set_axisbelow(True)

        axes[1].barh(no_dummy["Algorithm"], no_dummy["ROC-AUC"], color=colors, height=0.55)
        axes[1].set_xlabel("ROC-AUC Score")
        axes[1].set_title("ROC-AUC Comparison")
        axes[1].grid(axis='x', linestyle='--', alpha=0.7)
        axes[1].set_axisbelow(True)

        axes[2].barh(no_dummy["Algorithm"], no_dummy["PR-AUC"], color=colors, height=0.55)
        axes[2].set_xlabel("PR-AUC Score")
        axes[2].set_title("PR-AUC Comparison (Better for Imbalanced Data)")
        axes[2].axvline(0.1549, color='black', linestyle='--', lw=1, label="No-skill (0.155)")
        axes[2].grid(axis='x', linestyle='--', alpha=0.7)
        axes[2].set_axisbelow(True)
        axes[2].legend(loc="lower right")

        plt.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        st.markdown(
            '<div class="pastel-notice"><b>Metric Selection Note:</b> '
            'Weighted F1 barely separates the top 7 models (0.8799–0.8971). '
            'PR-AUC spreads them more clearly (0.6311–0.7458), making it the more useful metric '
            'for this 15.5% minority class. All models are well above the PR-AUC no-skill line of 0.1549. '
            'The ensembles beat the Review 1 Decision Tree only slightly on weighted F1 (+0.0040) '
            'but more on ranking (PR-AUC +0.0576).'
            '</div>',
            unsafe_allow_html=True,
        )

    with tab3:
        st.markdown('<p class="section-header" style="margin-top:0.5rem">Clustering Validation</p>', unsafe_allow_html=True)
        st.caption("Comparing MiniBatch K-Means and Agglomerative Hierarchical Clustering metrics.")

        st.markdown('<p class="section-header">MiniBatch K-Means (K=4)</p>', unsafe_allow_html=True)
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
            width='stretch', hide_index=True,
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
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        # Agglomerative Clustering Comparison
        st.markdown("<hr style='margin:1.5rem 0'>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Agglomerative Hierarchical Clustering (Ward, K=4)</p>', unsafe_allow_html=True)
        agg_comparison_data = load_agg_clustering_artifacts()
        if agg_comparison_data is None:
            st.caption("Agglomerative artifacts not found. Run the export cell in agg_clustering.ipynb.")
        else:
            agg_cmp_eval = agg_comparison_data["eval_metrics"]
            agg_cmp_config = agg_comparison_data["config"]
            st.caption(
                f"Metrics from {agg_cmp_config['hierarchical_sample_size']:,}-session fitted sample "
                f"with exact hierarchical labels. Uses 7 features (no max_price_viewed)."
            )
            agg_cmp_m1, agg_cmp_m2, agg_cmp_m3 = st.columns(3, gap="medium")
            agg_cmp_m1.markdown(
                f'<div class="metric-card"><div class="value">{agg_cmp_eval.get("Silhouette Score", 0):.4f}</div>'
                f'<div class="label">Silhouette Score</div></div>', unsafe_allow_html=True)
            agg_cmp_m2.markdown(
                f'<div class="metric-card"><div class="value">{agg_cmp_eval.get("Davies-Bouldin Index", 0):.4f}</div>'
                f'<div class="label">Davies-Bouldin Index</div></div>', unsafe_allow_html=True)
            agg_cmp_m3.markdown(
                f'<div class="metric-card"><div class="value">{agg_cmp_eval.get("Calinski-Harabasz Index", 0):,.2f}</div>'
                f'<div class="label">Calinski-Harabasz Index</div></div>', unsafe_allow_html=True)

            # K-sweep for Agglomerative
            agg_eval_k = agg_comparison_data["eval_by_k"].reset_index()
            agg_eval_k_display = agg_eval_k.copy()
            agg_eval_k_display["Selection"] = agg_eval_k_display["K"].apply(
                lambda k: f"K={k} (Selected)" if k == agg_cmp_config["n_clusters"] else ""
            )
            st.dataframe(
                agg_eval_k_display.style.format({
                    "Silhouette Score": "{:.4f}",
                    "Davies-Bouldin Index": "{:.4f}",
                    "Calinski-Harabasz Index": "{:,.2f}",
                    "Smallest Cluster (%)": "{:.1f}%",
                    "Largest Cluster (%)": "{:.1f}%",
                }),
                width='stretch', hide_index=True,
            )

            # Agglomerative K-selection visualizations
            st.markdown('<p class="section-header" style="margin-top:1rem">Agglomerative K-Selection Metrics</p>', unsafe_allow_html=True)
            agg_k_vals = agg_eval_k["K"].values
            selected_k = agg_cmp_config["n_clusters"]

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            # Silhouette Score vs K
            axes[0].plot(agg_k_vals, agg_eval_k["Silhouette Score"], "o-", color="#8B83C9", linewidth=2.0, markersize=6)
            axes[0].axvline(selected_k, linestyle="--", color="#A8C9B5", label=f"K={selected_k} Selected", linewidth=1.5)
            axes[0].set_xlabel("Number of Clusters (K)")
            axes[0].set_ylabel("Silhouette Score (Higher is Better)")
            axes[0].set_title("Silhouette Score vs K")
            axes[0].set_xticks(agg_k_vals)
            axes[0].grid(True, linestyle='--', alpha=0.7)
            axes[0].set_axisbelow(True)
            axes[0].legend(loc="upper right")

            # Davies-Bouldin Index vs K
            axes[1].plot(agg_k_vals, agg_eval_k["Davies-Bouldin Index"], "o-", color="#E8B9A8", linewidth=2.0, markersize=6)
            axes[1].axvline(selected_k, linestyle="--", color="#A8C9B5", label=f"K={selected_k} Selected", linewidth=1.5)
            axes[1].set_xlabel("Number of Clusters (K)")
            axes[1].set_ylabel("Davies-Bouldin Index (Lower is Better)")
            axes[1].set_title("Davies-Bouldin Index vs K")
            axes[1].set_xticks(agg_k_vals)
            axes[1].grid(True, linestyle='--', alpha=0.7)
            axes[1].set_axisbelow(True)
            axes[1].legend(loc="upper right")

            # Calinski-Harabasz Index vs K
            axes[2].plot(agg_k_vals, agg_eval_k["Calinski-Harabasz Index"], "o-", color="#A9B9E8", linewidth=2.0, markersize=6)
            axes[2].axvline(selected_k, linestyle="--", color="#A8C9B5", label=f"K={selected_k} Selected", linewidth=1.5)
            axes[2].set_xlabel("Number of Clusters (K)")
            axes[2].set_ylabel("Calinski-Harabasz Index (Higher is Better)")
            axes[2].set_title("Calinski-Harabasz Index vs K")
            axes[2].set_xticks(agg_k_vals)
            axes[2].grid(True, linestyle='--', alpha=0.7)
            axes[2].set_axisbelow(True)
            axes[2].legend(loc="upper right")

            plt.tight_layout()
            st.pyplot(fig, width='stretch')
            plt.close(fig)

            st.markdown(
                '<div class="pastel-notice"><b>Comparison note:</b> '
                'MiniBatch K-Means and Agglomerative metrics are <b>not directly comparable</b> — '
                'they use different evaluation sample sizes (50k vs 5k/2k), different feature sets '
                '(8 vs 7 features; K-Means includes max_price_viewed), and different evaluation '
                'protocols. Compare trends across K values within each algorithm rather than '
                'comparing absolute scores between algorithms.'
                '</div>',
                unsafe_allow_html=True,
            )


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
    <ul style="color:#626478;font-size:0.92rem;line-height:1.65;margin-bottom:0;">
        <li><b>Track 1 (Regression):</b> Predict daily aggregate sales volume in GBP (£) from order frequency, SKU breadth, and calendar signals to assist inventory management.</li>
        <li><b>Track 2 (Classification):</b> Identify session conversion likelihood (Revenue = True) to enable automated real-time discount incentives without wasting margin.</li>
        <li><b>Track 3 (Clustering):</b> Segment multi-million session catalogs into actionable user groups to distinguish window browsing, deep research, and high-intent buying.</li>
    </ul>
</div>

<div class="custom-card">
    <h3 style="margin-top:0;">Datasets & Scope</h3>
    <table style="width:100%;font-size:0.88rem;color:#303247;border-collapse:collapse;">
        <tr style="border-bottom:1px solid #E6E4EF;text-align:left;color:#626478;">
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
    <p style="font-size:0.92rem;color:#626478;line-height:1.65;margin:0;">
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
    <ul style="color:#626478;font-size:0.92rem;line-height:1.75;margin:0;padding-left:1.2rem;">
        <li><b>Python:</b> 3.12</li>
        <li><b>Streamlit:</b> 1.65+</li>
        <li><b>scikit-learn:</b> 1.8</li>
        <li><b>pandas:</b> 2.2</li>
        <li><b>numpy:</b> 2.0</li>
        <li><b>matplotlib:</b> 3.9</li>
        <li><b>joblib:</b> 1.5</li>
    </ul>
</div>
""", unsafe_allow_html=True)

        st.markdown('<p class="section-header" style="margin-top:0.5rem">Start Application</p>', unsafe_allow_html=True)
        st.code("streamlit run app.py", language="bash")
        st.markdown('<p class="section-header" style="margin-top:0.5rem">Export Models</p>', unsafe_allow_html=True)
        st.code("python export_models.py --task regression\npython export_models.py --task classification", language="bash")
