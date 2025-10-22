"""
AMaLWA Pro - Single-file Streamlit ML Studio (Identifier-safe)
Author: Dawood Junaid + ChatGPT (Senior Engineer style)
Note: identifier columns (Name/ID) are preserved for display and outputs but excluded
by default from model features. You can opt-in to include them as features.
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, r2_score, mean_squared_error, roc_curve
)

# ---------------------------
# Page config & simple styling
# ---------------------------
st.set_page_config(page_title="AMaLWA Pro", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
    .stButton>button {border-radius: 8px; padding: 6px 12px;}
    .reportview-container .main .block-container{padding-top:1rem;}
</style>""", unsafe_allow_html=True)

# ---------------------------
# Helper utilities
# ---------------------------
@st.cache_data
def load_demo_df(name: str):
    """Return demo dataset by name."""
    from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
    if name.startswith("Breast"):
        d = load_breast_cancer(as_frame=True)
        df = d.frame
    elif name.startswith("Iris"):
        d = load_iris(as_frame=True)
        df = d.frame
        # iris target is numeric but categorical-like
        df['target'] = df['target'].astype(int)
    else:
        d = load_diabetes(as_frame=True)
        df = d.frame
    return df

@st.cache_data
def df_to_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode('utf-8')

def model_to_download_bytes(obj):
    buf = BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    return buf.read()

def create_download_link(bytes_obj, filename, label):
    b64 = base64.b64encode(bytes_obj).decode()
    href = f"data:application/octet-stream;base64,{b64}"
    return f"<a href='{href}' download='{filename}'>{label}</a>"

def detect_task_type(y: pd.Series):
    """Return 'Classification' or 'Regression' by heuristic."""
    # numeric with many unique values -> regression
    if y.dtype.kind in 'biufc' and y.nunique() > 20:
        return 'Regression'
    return 'Classification'

def safe_import_shap():
    """Try to import shap; return module or None."""
    try:
        import shap
        return shap
    except Exception:
        return None

def detect_identifier_columns(df: pd.DataFrame):
    """
    Heuristic to detect identifier-like columns:
    - columns whose name contains keywords like 'name','id','player','bowler' etc.
    - OR object columns with unique values equal to number of rows (likely unique IDs)
    - OR object columns with very high cardinality ratio (e.g., > 0.9)
    Returns list of columns considered identifiers.
    """
    id_keywords = ['name','id','player','bowler','batsman','email','username','uid','guid','code','team']
    id_cols = []
    n = len(df)
    for c in df.columns:
        low = c.lower()
        if any(kw in low for kw in id_keywords):
            id_cols.append(c)
            continue
        if df[c].dtype == object:
            try:
                nunique = df[c].nunique(dropna=True)
                if nunique == n and n > 10:  # unique per row
                    id_cols.append(c)
                elif n>0 and (nunique / n) > 0.9 and n > 50:  # very high cardinality
                    id_cols.append(c)
            except Exception:
                pass
    # ensure uniqueness
    id_cols = sorted(list(set(id_cols)))
    return id_cols

# ---------------------------
# Session initialization
# ---------------------------
if 'model' not in st.session_state:
    st.session_state.model = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'X_cols' not in st.session_state:
    st.session_state.X_cols = None
if 'detected_task' not in st.session_state:
    st.session_state.detected_task = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'id_cols' not in st.session_state:
    st.session_state.id_cols = []

# ---------------------------
# Sidebar - configuration
# ---------------------------
st.sidebar.header("AMaLWA Pro — Configuration")

uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=['csv'])
use_demo = st.sidebar.checkbox("Use demo dataset", value=True)
demo_choice = st.sidebar.selectbox("Demo dataset", ["Breast Cancer (Classification)", "Iris (Classification)", "Diabetes (Regression)"], index=0)
if st.sidebar.button("Reload demo"):
    st.rerun()

st.sidebar.markdown("---")
task_choice = st.sidebar.selectbox("Force task type (optional)", ["Auto-detect", "Classification", "Regression"], index=0)
algo_choice = st.sidebar.selectbox("Algorithm", [
    'Auto (recommended)', 'LogisticRegression', 'RandomForest', 'KNN', 'NaiveBayes', 'SVM', 'LinearRegression', 'RandomForestRegressor'
], index=0)
test_pct = st.sidebar.slider("Test set size (%)", 10, 50, 20)
random_state = int(st.sidebar.number_input("Random State", min_value=0, value=42, step=1))
scale_numeric = st.sidebar.checkbox("Scale numeric features", value=True)
cross_val = st.sidebar.checkbox("Cross-validate (5-fold)", value=False)
st.sidebar.markdown("---")
st.sidebar.header("Hyperparameters (quick)")
n_estimators = st.sidebar.slider("n_estimators (RF)", 10, 300, 100)
max_depth = st.sidebar.slider("max_depth (RF)", 1, 50, 6)
k_neighbors = st.sidebar.slider("n_neighbors (KNN)", 1, 31, 5)
c_value = st.sidebar.slider("C (SVM/LogReg)", 0.01, 10.0, 1.0)
st.sidebar.markdown("---")
shap_toggle = st.sidebar.checkbox("Enable SHAP explainability (optional)", value=False)
st.sidebar.caption("If SHAP isn't installed the app will suggest how to install it.")

# ---------------------------
# Load dataframe
# ---------------------------
if uploaded_file is not None and not use_demo:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Uploaded CSV loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
        df = pd.DataFrame()
else:
    df = load_demo_df(demo_choice)
    st.sidebar.info(f"Loaded demo: {demo_choice}")

# Quick dataset info
st.title("🤖 AMaLWA Pro — Automated ML Studio (Identifier-safe)")
st.markdown("Upload a CSV or use a demo dataset. Configure preprocessing, choose algorithm, train, and download the model/predictions.")

st.subheader("Dataset preview")
col_a, col_b = st.columns([3,1])
with col_a:
    st.dataframe(df.head(200))
with col_b:
    st.write("Shape:")
    st.write(df.shape)
    if st.button("Download dataset CSV"):
        bytes_csv = df_to_csv_bytes(df)
        st.markdown(create_download_link(bytes_csv, "dataset.csv", "Download dataset (CSV)"), unsafe_allow_html=True)

# ---------------------------
# Cleaning controls
# ---------------------------
st.subheader("Data cleaning & conversions")
with st.expander("Cleaning controls (click to expand)"):
    drop_na = st.checkbox("Drop rows with any NA", value=False)
    fill_na = st.checkbox("Fill NA (numeric->median, categorical->mode)", value=True)
    convert_types = st.checkbox("Try convert object columns to numeric when possible", value=True)
if drop_na:
    df = df.dropna()
    st.write("Dropped NA rows — new shape:", df.shape)
elif fill_na:
    for c in df.columns:
        if df[c].dtype.kind in 'biufc':
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else '')
    st.write("Filled NA values where applicable.")

if convert_types:
    for c in df.select_dtypes(include=['object']).columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass

# ---------------------------
# Detect identifier columns (preserve them)
# ---------------------------
id_cols_detected = detect_identifier_columns(df)
st.session_state.id_cols = id_cols_detected  # store for later use
if id_cols_detected:
    st.info(f"Detected identifier columns (kept for display/export, excluded from features by default): {id_cols_detected}")
else:
    st.info("No obvious identifier columns detected.")

# ---------------------------
# Modeling setup
# ---------------------------
st.subheader("Model setup")
if df.empty:
    st.warning("No dataset available. Upload a CSV or enable demo dataset.")
    st.stop()

all_columns = list(df.columns)
target_col = st.selectbox("Select target column (what you want to predict)", all_columns, index=len(all_columns)-1)

# Build default feature pool: all columns except target and id columns
default_feature_pool = [c for c in all_columns if c != target_col and c not in id_cols_detected]

# Provide user option to include id columns as features
include_id_as_features = st.checkbox("Include detected identifier columns as features (not recommended)", value=False)
if include_id_as_features:
    # let user choose which id cols to include
    include_id_choice = st.multiselect("Select identifier columns to include as features", id_cols_detected, default=id_cols_detected)
    # merge pools
    feature_pool = default_feature_pool + include_id_choice
else:
    feature_pool = default_feature_pool

feature_cols = st.multiselect("Feature columns (optional — leave empty to use full feature pool)", feature_pool, default=feature_pool)
if not feature_cols:
    X = df[[c for c in feature_pool]].copy()
else:
    X = df[feature_cols].copy()

# Keep an identifier dataframe for later mapping (display/download)
id_df = df[id_cols_detected].copy() if id_cols_detected else pd.DataFrame(index=df.index)
y = df[target_col].copy()

# Detect task
if task_choice == 'Auto-detect':
    detected_task = detect_task_type(y)
else:
    detected_task = task_choice
st.write(f"Detected task: **{detected_task}**")
st.session_state.detected_task = detected_task

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
st.write(f"Numeric features: {num_features}")
st.write(f"Categorical features: {cat_features}")

# ---------------------------
# Preprocessor pipeline
# ---------------------------
num_transform = [('imputer', SimpleImputer(strategy='median'))]
if scale_numeric:
    num_transform.append(('scaler', StandardScaler()))

num_pipeline = Pipeline(num_transform)
# Use sparse_output=False for sklearn >=1.2; older versions may use sparse=False
try:
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                             ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
except TypeError:
    # fallback for older sklearn versions
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                             ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

transformers = []
if num_features:
    transformers.append(('num', num_pipeline, num_features))
if cat_features:
    transformers.append(('cat', cat_pipeline, cat_features))

preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

# ---------------------------
# Algorithm selection mapping
# ---------------------------
def choose_model(algo_name, task):
    """Return a sklearn estimator based on choice and detected task."""
    if algo_name == 'Auto (recommended)':
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state) if task == 'Classification' else RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    if algo_name == 'LogisticRegression':
        return LogisticRegression(C=c_value, max_iter=1000) if task == 'Classification' else LinearRegression()
    if algo_name == 'RandomForest':
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state) if task == 'Classification' else RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    if algo_name == 'KNN':
        return KNeighborsClassifier(n_neighbors=k_neighbors) if task == 'Classification' else KNeighborsClassifier(n_neighbors=k_neighbors)
    if algo_name == 'NaiveBayes':
        return GaussianNB()
    if algo_name == 'SVM':
        return SVC(C=c_value, probability=True) if task == 'Classification' else SVR(C=c_value)
    if algo_name == 'LinearRegression':
        return LinearRegression()
    if algo_name == 'RandomForestRegressor':
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    # fallback
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state) if task == 'Classification' else RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

model_estimator = choose_model(algo_choice, detected_task)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_estimator)])

# ---------------------------
# Train the model
# ---------------------------
st.subheader("Train & Evaluate")
train_col, eval_col = st.columns([1,2])

with train_col:
    train_button = st.button("🚀 Train Model")
    if train_button:
        # train
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct/100.0, random_state=random_state)
            t0 = time.time()
            with st.spinner("Training model..."):
                pipeline.fit(X_train, y_train)
            train_time = time.time() - t0
            st.success(f"Model trained in {train_time:.2f}s")
            st.session_state.pipeline = pipeline
            st.session_state.model = pipeline.named_steps['model']
            st.session_state.X_cols = X.columns.tolist()
            st.session_state.id_cols = id_cols_detected  # persist detected ids

            # cross-validate
            if cross_val:
                scoring = 'accuracy' if detected_task == 'Classification' else 'r2'
                with st.spinner("Performing cross-validation..."):
                    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)
                st.write(f"5-fold CV ({scoring}) mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                st.session_state.metrics = {'cv_scores': cv_scores}

            # predictions & metrics
            y_pred = pipeline.predict(X_test)
            metrics = {}
            if detected_task == 'Classification':
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                st.write("**Classification metrics:**")
                st.write(f"Accuracy: {metrics['accuracy']:.4f}")
                st.write(f"Precision (weighted): {metrics['precision']:.4f}")
                st.write(f"Recall (weighted): {metrics['recall']:.4f}")
                st.write(f"F1 (weighted): {metrics['f1']:.4f}")

                # confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(5,4))
                sns.heatmap(cm, annot=True, fmt='d', ax=ax_cm)
                ax_cm.set_ylabel('Actual')
                ax_cm.set_xlabel('Predicted')
                st.pyplot(fig_cm)

                # ROC AUC if binary
                if len(np.unique(y_test)) == 2 and hasattr(pipeline.named_steps['model'], 'predict_proba'):
                    y_proba = pipeline.predict_proba(X_test)[:,1]
                    auc = roc_auc_score(y_test, y_proba)
                    st.write(f"ROC AUC (binary): {auc:.4f}")
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    fig_roc, ax_roc = plt.subplots()
                    ax_roc.plot(fpr, tpr, label=f"AUC={auc:.3f}")
                    ax_roc.plot([0,1],[0,1],'--', color='gray')
                    ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR'); ax_roc.legend()
                    st.pyplot(fig_roc)
            else:
                metrics['r2'] = r2_score(y_test, y_pred)
                metrics['mse'] = mean_squared_error(y_test, y_pred)
                st.write("**Regression metrics:**")
                st.write(f"R²: {metrics['r2']:.4f}")
                st.write(f"MSE: {metrics['mse']:.4f}")

            # store metrics and last test split (for predict explanations)
            st.session_state.metrics = metrics
            st.session_state['last_test'] = {'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred}
        except Exception as e:
            st.error(f"Training failed: {e}")

with eval_col:
    st.subheader("Model Insights & Export")
    if st.session_state.pipeline is None:
        st.info("No trained model yet. Train a model to see insights and export options.")
    else:
        # Feature importance / coefficients
        try:
            model_obj = st.session_state.pipeline.named_steps['model']
            preproc_obj = st.session_state.pipeline.named_steps['preprocessor']
            st.write("**Feature importance / coefficients**")
            # attempt to get feature names produced by preprocessor
            feature_names = []
            if num_features:
                feature_names.extend(num_features)
            if cat_features:
                # get onehot feature names if present
                try:
                    ohe = preproc_obj.named_transformers_['cat'].named_steps['onehot']
                    cat_names = ohe.get_feature_names_out(cat_features).tolist()
                    feature_names.extend(cat_names)
                except Exception:
                    # fallback to raw cat column names
                    feature_names.extend(cat_features)
            # tree-based feature importance
            if hasattr(model_obj, 'feature_importances_'):
                importances = model_obj.feature_importances_
                fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(30)
                st.bar_chart(fi)
            elif hasattr(model_obj, 'coef_'):
                coefs = model_obj.coef_
                if coefs.ndim > 1:
                    coefs = coefs.mean(axis=0)
                coef_series = pd.Series(coefs, index=feature_names).sort_values(key=abs, ascending=False).head(30)
                st.write(coef_series)
            else:
                st.write("Model does not expose importances/coefficients.")
        except Exception as e:
            st.write("Could not compute importance/coefficients:", e)

        # SHAP explainability (optional)
        if shap_toggle:
            st.write("---")
            st.write("**SHAP Explainability**")
            shap_mod = safe_import_shap()
            if shap_mod is None:
                st.warning("SHAP is not installed. Install with `pip install shap` to enable detailed explanations.")
            else:
                try:
                    explainer = None
                    model_obj = st.session_state.pipeline.named_steps['model']
                    X_test = st.session_state['last_test']['X_test']
                    # build explainer on transformed data for speed
                    transformed = st.session_state.pipeline.named_steps['preprocessor'].transform(X_test)
                    explainer = shap_mod.Explainer(model_obj, transformed)
                    shap_values = explainer(transformed)
                    st.write("SHAP summary plot (may take a moment)...")
                    shap_mod.plots.beeswarm(shap_values, show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()
                except Exception as e:
                    st.write("SHAP explanation failed:", e)

        # Export trained model
        st.write("---")
        st.write("**Export**")
        bytes_model = model_to_download_bytes(st.session_state.pipeline)
        st.markdown(create_download_link(bytes_model, "trained_pipeline.pkl", "Download trained pipeline (.pkl)"), unsafe_allow_html=True)

        # Simple text report
        if st.button("Download performance report (TXT)"):
            rep_lines = []
            rep_lines.append("AMaLWA Pro - Performance Report")
            rep_lines.append(f"Task: {st.session_state.detected_task}")
            rep_lines.append(f"Algorithm: {type(st.session_state.pipeline.named_steps['model']).__name__}")
            if st.session_state.metrics:
                for k,v in st.session_state.metrics.items():
                    rep_lines.append(f"{k}: {v}")
            report_bytes = "\n".join(map(str, rep_lines)).encode('utf-8')
            st.markdown(create_download_link(report_bytes, "performance_report.txt", "Download report (.txt)"), unsafe_allow_html=True)

# ---------------------------
# Batch prediction upload & single-row predict
# ---------------------------
st.subheader("Predictions")
pred_col1, pred_col2 = st.columns([1,1])

with pred_col1:
    st.write("**Single-sample prediction**")
    if st.session_state.pipeline is None:
        st.info("Train a model to enable predictions.")
    else:
        input_vals = {}
        id_inputs = {}
        with st.form("single_predict_form"):
            # optionally collect identifier for the single sample so we can show it in the result
            if st.session_state.id_cols:
                st.write("Identifier fields (optional, for mapping output):")
                for idc in st.session_state.id_cols:
                    id_inputs[idc] = st.text_input(f"{idc}", value="")
            st.write("Feature inputs:")
            for c in X.columns:
                if c in num_features:
                    input_vals[c] = st.number_input(f"{c}", value=float(X[c].median()))
                else:
                    uniques = list(X[c].dropna().unique())[:50]
                    if len(uniques) <= 20 and len(uniques)>0:
                        input_vals[c] = st.selectbox(c, uniques)
                    else:
                        input_vals[c] = st.text_input(c, value='')
            submitted = st.form_submit_button("Predict")
            if submitted:
                try:
                    df_in = pd.DataFrame([input_vals])
                    pred = st.session_state.pipeline.predict(df_in)
                    out_text = f"Prediction: {pred[0]}"
                    # show id mapping if provided
                    if id_inputs:
                        mapping = {k: v for k,v in id_inputs.items() if v}
                        if mapping:
                            out_text += " | " + ", ".join([f"{k}={v}" for k,v in mapping.items()])
                    st.success(out_text)
                    if st.session_state.detected_task == 'Classification' and hasattr(st.session_state.pipeline.named_steps['model'], 'predict_proba'):
                        probs = st.session_state.pipeline.predict_proba(df_in)
                        st.write("Class probabilities:", probs.tolist())
                except Exception as e:
                    st.error("Prediction failed: " + str(e))

with pred_col2:
    st.write("**Batch prediction (CSV)**")
    batch_file = st.file_uploader("Upload CSV for batch predictions", type=['csv'], key="batch_pred")
    if st.button("Run batch predictions") and batch_file is not None:
        if st.session_state.pipeline is None:
            st.error("Train a model first.")
        else:
            try:
                df_batch = pd.read_csv(batch_file)
                # Keep any identifier columns if present in uploaded file
                batch_id_cols = [c for c in st.session_state.id_cols if c in df_batch.columns]
                # ensure all required features exist
                missing_cols = [c for c in st.session_state.X_cols if c not in df_batch.columns]
                if missing_cols:
                    st.error(f"Uploaded CSV missing required feature columns: {missing_cols}")
                else:
                    X_batch = df_batch[st.session_state.X_cols]
                    preds = st.session_state.pipeline.predict(X_batch)
                    df_batch['prediction'] = preds
                    # if training id columns are present, keep them in output for mapping
                    if batch_id_cols:
                        display_cols = batch_id_cols + st.session_state.X_cols + ['prediction']
                    else:
                        display_cols = st.session_state.X_cols + ['prediction']
                    st.dataframe(df_batch[display_cols].head(200))
                    csv_pred = df_batch.to_csv(index=False).encode('utf-8')
                    st.markdown(create_download_link(csv_pred, "predictions.csv", "Download predictions (CSV)"), unsafe_allow_html=True)
            except Exception as e:
                st.error("Batch prediction failed: " + str(e))

# ---------------------------
# Quick EDA tools
# ---------------------------
st.subheader("Quick EDA & Visuals")
eda1, eda2 = st.columns([1,1])
with eda1:
    if st.checkbox("Show dataset summary"):
        st.write(df.describe(include='all'))
    if st.checkbox("Show numeric histograms"):
        if len(num_features) == 0:
            st.info("No numeric features to show histograms.")
        else:
            fig_hist, axs = plt.subplots(nrows=min(4,len(num_features)), ncols=1, figsize=(6, 3*min(4,len(num_features))))
            for i,c in enumerate(num_features[:4]):
                axs[i].hist(df[c].dropna(), bins=30)
                axs[i].set_title(c)
            st.pyplot(fig_hist)
with eda2:
    if st.checkbox("Show correlation heatmap"):
        corr = df.select_dtypes(include=[np.number]).corr()
        fig_corr, ax_corr = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

st.markdown("---")
st.markdown("""
### Made with ❤️ — AMaLWA Pro  
By:<br>
**Muhammad Hammad Shah**<br>
**Muhammad Usman Khan**<br>
**Sharjeel Majeed**
""", unsafe_allow_html=True)
