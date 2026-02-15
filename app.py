import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.set_page_config(page_title="Week 5 - Work-from-Home Burnout Analysis", layout="wide")

DATASET_ID = "sonalshinde123/work-from-home-employee-burnout-dataset"


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    app_dir = os.path.dirname(__file__)
    csv_path = os.path.join(app_dir, "work_from_home_burnout_dataset.csv")
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        st.error(f"Dataset file not found: {csv_path}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        raise


def choose_targets(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # For burnout dataset: use burnout_score for regression, burnout_risk for classification
    regression_target = "burnout_score" if "burnout_score" in df.columns else numeric_cols[-1]
    classification_target = "burnout_risk"  # Fixed to always use burnout_risk to match Problem Definition

    work_df = df.copy()

    return work_df, regression_target, classification_target


def split_features_target(df: pd.DataFrame, target: str):
    # Drop target and non-feature columns (user_id, task_completion_rate to avoid data leakage)
    exclude_cols = [target, "user_id", "task_completion_rate"]
    cols_to_drop = [c for c in exclude_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target]
    return X, y


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_regression(df: pd.DataFrame, target: str):
    X, y = split_features_target(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor_for_lr = build_preprocessor(X_train, scale_numeric=False)
    preprocessor_for_rf = build_preprocessor(X_train, scale_numeric=False)

    models = {
        "Baseline (mean)": Pipeline(
            steps=[("preprocessor", preprocessor_for_lr), ("model", DummyRegressor(strategy="mean"))]
        ),
        "Linear Regression": Pipeline(
            steps=[("preprocessor", preprocessor_for_lr), ("model", LinearRegression())]
        ),
        "Random Forest Regressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor_for_rf),
                (
                    "model",
                    RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42),
                ),
            ]
        ),
    }

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions[name] = preds
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, preds),
                "RMSE": rmse,
                "R2": r2_score(y_test, preds),
            }
        )

    # 5-fold cross-validation for one model
    cv_model = Pipeline(
        steps=[("preprocessor", preprocessor_for_lr), ("model", LinearRegression())]
    )
    # Use only the training split for cross-validation to avoid leaking test data into CV
    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")

    return pd.DataFrame(results), y_test, predictions, cv_scores


def evaluate_classification(df: pd.DataFrame, target: str):
    X, y = split_features_target(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor_scaled = build_preprocessor(X_train, scale_numeric=True)
    preprocessor_tree = build_preprocessor(X_train, scale_numeric=False)

    models = {
        "Baseline (majority)": Pipeline(
            steps=[("preprocessor", preprocessor_tree), ("model", DummyClassifier(strategy="most_frequent"))]
        ),
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor_scaled),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "Random Forest Classifier": Pipeline(
            steps=[
                ("preprocessor", preprocessor_tree),
                (
                    "model",
                    RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced"),
                ),
            ]
        ),
    }

    results = []
    confusion_data = {}
    per_class_recall = {}

    average_mode = "macro"  # Use macro-F1 for imbalanced classes (not weighted)

    # Fixed label order for metrics/plots
    labels = np.unique(y)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds, labels=labels)
        confusion_data[name] = cm

        # Per-class recall: proportion of each true class correctly identified
        per_class_recall[name] = recall_score(y_test, preds, average=None, labels=labels, zero_division=0)

        # Compute macro and per-class metrics for imbalanced classification
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision (macro)": precision_score(y_test, preds, average="macro", zero_division=0),
                "Recall (macro)": recall_score(y_test, preds, average="macro", zero_division=0),
                "F1 (macro)": f1_score(y_test, preds, average="macro", zero_division=0),
            }
        )

    # Convert per-class recall to a tidy DataFrame (rows=models, columns=class labels)
    recall_df = pd.DataFrame(per_class_recall, index=labels).T.reset_index().rename(columns={'index': 'Model'})
    return pd.DataFrame(results), y_test, confusion_data, recall_df, labels


def regression_plot(y_test, y_pred, model_name: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, s=50, color='steelblue')
    min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="Perfect prediction")
    ax.set_title(f"Predicted vs Actual ({model_name})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Actual", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def confusion_plot(cm: np.ndarray, labels, model_name: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    # Use matplotlib only (no seaborn dependency)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix ({model_name})", fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontweight='bold')
    
    plt.colorbar(im, ax=ax, label="Count")
    return fig


def main():
    st.title("📊 Week 5: Supervised Learning (Work-from-Home Burnout Analysis)")

    # Table of Contents
    st.markdown("""
    ---
    **📑 Quick Navigation:**
    - [Data & EDA](#a-data--eda) | [Modeling](#b-modeling) | [Visualizations](#c-visualizations) | [Documentation](#d-documentation)
    
    ---
    """)

    st.markdown(
        """
### Problem Definition
- **Regression:** Predict numeric `burnout_score` and achieve better results than predicting the mean value.
- **Classification:** Predict categorical `burnout_risk` and achieve better results than the majority class baseline.

These predictions support decision-making for employee wellness interventions and burnout prevention strategies.
        """
    )
    
    with st.expander("ℹ️ **About the dataset and class imbalance** (click to expand)", expanded=False):
        st.markdown("""
**`burnout_risk` is derived by binning `burnout_score`:**
- **Low** (approx. burnout_score < 70): 1527 samples (84.8%)
- **Medium** (approx. 70–110): 253 samples (14.1%)
- **High** (approx. >110): 20 samples (1.1%)

⚠️ **Class imbalance impact:** Accuracy is misleading. Reported metrics emphasize **macro-F1** and **per-class recall**. Classifiers use `class_weight='balanced'` to prevent ignoring minority classes.
        """)

    # Helper function to style best result rows
    def highlight_best_regression(df):
        """Highlight the row with minimum RMSE"""
        result = df.copy()
        if "RMSE" in result.columns:
            min_rmse_idx = result["RMSE"].idxmin()
            for col in result.columns:
                result.loc[min_rmse_idx, col] = f"**{result.loc[min_rmse_idx, col]}**"
        return result

    def highlight_best_classification(df):
        """Highlight the row with maximum F1 (macro)"""
        result = df.copy()
        if "F1 (macro)" in result.columns:
            max_f1_idx = result["F1 (macro)"].idxmax()
            for col in result.columns:
                result.loc[max_f1_idx, col] = f"**{result.loc[max_f1_idx, col]}**"
        return result

    try:
        df_raw = load_data()
    except Exception:
        st.error("Failed to load dataset. A local file can be provided (.csv/.parquet/.xlsx).")
        uploaded = st.file_uploader("Dataset file uploader", type=["csv", "parquet", "xlsx"])
        if uploaded is None:
            st.stop()
        else:
            try:
                name = uploaded.name.lower()
                if name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded)
                elif name.endswith(".parquet"):
                    df_raw = pd.read_parquet(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded)
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
                st.stop()

    df, reg_target, cls_target = choose_targets(df_raw)

    st.header("A) Data & EDA")
    with st.expander("📈 **Dataset Overview**", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values (total)", int(df.isna().sum().sum()))

        st.subheader("Head of data")
        st.dataframe(df.head(), use_container_width=True)

        col_stats, col_dist = st.columns([1, 1])
        with col_stats:
            st.subheader("Summary Statistics")
            st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
        
        with col_dist:
            st.subheader("Distribution Plot")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            hist_col = st.selectbox("Select variable for histogram", numeric_cols, key="hist_select")
            fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
            ax_hist.hist(df[hist_col].dropna(), bins=25, color='steelblue', edgecolor='black', alpha=0.7)
            ax_hist.set_title(f"Histogram: {hist_col}", fontsize=12, fontweight='bold')
            ax_hist.set_xlabel(hist_col, fontsize=10)
            ax_hist.set_ylabel("Count", fontsize=10)
            ax_hist.grid(True, alpha=0.3)
            st.pyplot(fig_hist)

    st.header("B) Modeling")

    with st.expander("⚙️ **Model Settings & Rationale**", expanded=True):
        model_settings = [
            {"Model": "Baseline (mean)", "Key hyperparameters": "strategy='mean'"},
            {"Model": "Linear Regression", "Key hyperparameters": "ordinary least squares (default)"},
            {"Model": "Random Forest Regressor", "Key hyperparameters": "n_estimators=300, max_depth=12, random_state=42"},
            {"Model": "Baseline (majority)", "Key hyperparameters": "strategy='most_frequent'"},
            {"Model": "Logistic Regression", "Key hyperparameters": "max_iter=2000, class_weight='balanced'"},
            {"Model": "Random Forest Classifier", "Key hyperparameters": "n_estimators=300, max_depth=12, class_weight='balanced', random_state=42"},
        ]
        st.table(pd.DataFrame(model_settings))
        st.markdown("""
**Why these settings were chosen:**
- **Baseline models** use default strategies (mean/majority) to set a minimum performance bar.
- **Linear Regression** uses ordinary least squares (simple, interpretable, fast) to establish a linear baseline.
- **Random Forest** (`n_estimators=300, max_depth=12`): 300 trees balance pattern capture with computational efficiency; max_depth=12 prevents overfitting on this 1800-sample dataset and protects against memorizing noise in rare High-class samples.
- **Logistic Regression** (`max_iter=2000`): higher iteration limit ensures convergence; `class_weight='balanced'` addresses severe class imbalance.
- **class_weight='balanced'** (classifier): automatically upweights minority classes (High: 20 samples) to prevent the model from ignoring them.
- **random_state=42**: fixes randomness for reproducibility.

These are conservative, interpretable choices suitable for small imbalanced datasets.
        """)

    st.subheader("🔢 Regression Analysis")
    with st.expander("ℹ️ **Regression Setup**", expanded=False):
        st.write(
            "**Target:** " + reg_target + "\n\n" +
            "**Features:** day_type, work_hours, screen_time_hours, meetings_count, breaks_taken, after_hours_work, sleep_hours " +
            "(user_id and task_completion_rate excluded). Selected features cover work patterns (hours, screen time, meetings), " +
            "recovery (breaks, sleep), and work boundaries (after-hours work, day type) to capture burnout risk holistically."
        )
    
    reg_results, reg_y_test, reg_predictions, cv_scores = evaluate_regression(df, reg_target)
    # Display regression results with best row highlighted
    reg_sorted = reg_results.sort_values("RMSE")
    
    col_results, col_cv = st.columns([2, 1])
    with col_results:
        st.write("**Model Performance:**")
        st.dataframe(
            reg_sorted.style.highlight_min(subset=['RMSE'], color='#90EE90'),
            use_container_width=True
        )
    
    with col_cv:
        st.metric("5-fold CV RMSE (Linear Reg.)", f"{-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    best_reg_row = reg_results.sort_values("RMSE").iloc[0]
    st.success(
        f"✅ Best regression model: {best_reg_row['Model']} (RMSE={best_reg_row['RMSE']:.3f}, MAE={best_reg_row['MAE']:.3f})"
    )

    st.subheader("📊 Classification Analysis")
    with st.expander("ℹ️ **Classification Setup**", expanded=False):
        st.write(
            "**Target:** " + cls_target + "\n\n" +
            "**Features:** day_type, work_hours, screen_time_hours, meetings_count, breaks_taken, after_hours_work, sleep_hours " +
            "(user_id and task_completion_rate excluded to avoid data leakage). Same features as regression—work load, screen engagement, " +
            "recovery behaviors, and sleep quality are strong indicators of burnout risk."
        )
        st.warning("⚠️ **High class has only 20 samples** — results may vary substantially. Stratified split and class_weight='balanced' help, but interpret with caution.")
    
    cls_results, cls_y_test, confusion_data, cls_recall_df, cls_labels = evaluate_classification(df, cls_target)
    # Display classification results with best row highlighted
    cls_sorted = cls_results.sort_values("F1 (macro)", ascending=False)
    
    col_cls_results, col_recall = st.columns([1, 1])
    with col_cls_results:
        st.write("**Model Performance (F1-focused):**")
        st.dataframe(
            cls_sorted.style.highlight_max(subset=['F1 (macro)'], color='#90EE90'),
            use_container_width=True
        )
    
    with col_recall:
        st.write("**Per-class Recall (detection rate per class):**")
        st.dataframe(cls_recall_df.set_index("Model").round(3), use_container_width=True)

    # Highlight recall for the rare 'High' class if present
    if "High" in list(cls_labels):
        best_high = cls_recall_df.sort_values("High", ascending=False).iloc[0]
        st.warning(f"⚠️ **Best model for detecting 'High' risk:** {best_high['Model']} (High recall={best_high['High']:.3f})")

    class_distribution = df[cls_target].value_counts(normalize=True).rename("ratio").reset_index()
    st.info(f"**Class distribution:** Low: {class_distribution[class_distribution[cls_target]=='Low']['ratio'].values[0]*100:.1f}% | "
            f"Medium: {class_distribution[class_distribution[cls_target]=='Medium']['ratio'].values[0]*100:.1f}% | "
            f"High: {class_distribution[class_distribution[cls_target]=='High']['ratio'].values[0]*100:.1f}%")

    best_cls_row = cls_results.sort_values("F1 (macro)", ascending=False).iloc[0]
    st.success(
        f"✅ Best classification model: {best_cls_row['Model']} (Accuracy={best_cls_row['Accuracy']:.3f}, F1 (macro)={best_cls_row['F1 (macro)']:.3f})"
    )

    st.header("C) Visualizations")
    
    st.subheader("📈 Regression Predictions")
    reg_plot_model = st.selectbox("Regression prediction plot model", list(reg_predictions.keys()), index=1, key="reg_plot")
    st.pyplot(regression_plot(reg_y_test, reg_predictions[reg_plot_model], reg_plot_model))
    st.caption("📍 **Interpretation:** The closer the points are to the red dashed line, the better the model. Points above the line overpredict; points below underpredict.")

    st.subheader("🎯 Classification Confusion Matrices")
    cm_model = st.selectbox("Confusion matrix model", list(confusion_data.keys()), index=1, key="cm_model")
    st.pyplot(confusion_plot(confusion_data[cm_model], labels=cls_labels, model_name=cm_model))
    st.caption("📍 **Interpretation:** Diagonal cells show correct predictions (True Positives & True Negatives). Off-diagonal cells show misclassifications.")

    st.header("D) Documentation")
    
    with st.expander("📚 **Key Concepts**", expanded=False):
        st.markdown("""
- **Supervised learning:** Models the relationship between input features and a target variable using labeled training data.
- **Regression vs classification in this dataset:** Regression predicts the numeric `burnout_score`, while classification predicts the categorical `burnout_risk` (Low/Medium/High).
- **Features:** Selected features are day_type, work_hours, screen_time_hours, meetings_count, breaks_taken, after_hours_work, sleep_hours. Excluded: user_id (identifier only) and task_completion_rate (to avoid data leakage).
- **Class imbalance:** Burnout_risk has extreme imbalance (Low: 85%, Medium: 14%, High: 1%). class_weight='balanced' is used and evaluation emphasizes macro-F1 instead of accuracy.
- **Models:** Baseline (majority/mean) + Linear/Logistic regression + Random Forest (comparing simple vs non-linear approaches).
- **Conclusion:** A practical model is one that clearly outperforms the baseline and remains stable across cross-validation folds, with good macro-F1 scores on minority classes.
        """)
    
    with st.expander("🔍 **Model Explanations**", expanded=False):
        st.markdown("""
**Baseline (mean/majority):** Always predicts the training set average. Sets the minimum bar—if a model does not beat this significantly, it is not learning from the features.

**Linear Regression:** Assumes y ≈ b₀ + b₁x₁ + b₂x₂ + ... Straightforward, interpretable, fast. Works well if relationships are roughly linear.

**Random Forest:** Builds many decision trees and averages their predictions. Captures non-linear patterns and interactions without much tuning. Harder to interpret but often outperforms linear models.
        """)


if __name__ == "__main__":
    main()
