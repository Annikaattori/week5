import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub
from kagglehub import KaggleDatasetAdapter

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

st.set_page_config(page_title="Week 5 - Student Habits ML", layout="wide")

DATASET_ID = "jayaantanaath/student-habits-vs-academic-performance"


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    file_path = ""
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET_ID,
        file_path,
    )
    return df


def choose_targets(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    regression_target = "exam_score" if "exam_score" in df.columns else numeric_cols[-1]

    if "performance_level" in df.columns:
        classification_target = "performance_level"
        work_df = df.copy()
    else:
        # Binaus/luokittelu numeromuuttujasta
        q1, q2 = df[regression_target].quantile([0.33, 0.66])
        work_df = df.copy()
        work_df["performance_level"] = pd.cut(
            work_df[regression_target],
            bins=[-np.inf, q1, q2, np.inf],
            labels=["Low", "Medium", "High"],
        )
        classification_target = "performance_level"

    return work_df, regression_target, classification_target


def split_features_target(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
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
        "Baseline (mean)": DummyRegressor(strategy="mean"),
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

    # 5-fold CV yhdelle mallille
    cv_model = Pipeline(
        steps=[("preprocessor", preprocessor_for_lr), ("model", LinearRegression())]
    )
    cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring="neg_root_mean_squared_error")

    return pd.DataFrame(results), y_test, predictions, cv_scores


def evaluate_classification(df: pd.DataFrame, target: str):
    X, y = split_features_target(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor_scaled = build_preprocessor(X_train, scale_numeric=True)
    preprocessor_tree = build_preprocessor(X_train, scale_numeric=False)

    models = {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor_scaled),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
        "Random Forest Classifier": Pipeline(
            steps=[
                ("preprocessor", preprocessor_tree),
                (
                    "model",
                    RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42),
                ),
            ]
        ),
    }

    results = []
    confusion_data = {}

    average_mode = "binary" if y.nunique() == 2 else "weighted"

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds, labels=np.unique(y))
        confusion_data[name] = cm

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds, average=average_mode, zero_division=0),
                "Recall": recall_score(y_test, preds, average=average_mode, zero_division=0),
                "F1": f1_score(y_test, preds, average=average_mode, zero_division=0),
            }
        )

    return pd.DataFrame(results), y_test, confusion_data


def regression_plot(y_test, y_pred, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.7)
    min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], "r--", label="45° line")
    ax.set_title(f"Predicted vs Actual ({model_name})")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()
    return fig


def confusion_plot(cm: np.ndarray, labels, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix ({model_name})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


def main():
    st.title("Week 5: Supervised Learning (Student Habits Dataset)")

    st.markdown(
        """
### Problem definition
- **Regressio:** ennustetaan numeerinen `exam_score` ja pyritään parempaan tulokseen kuin aina keskiarvo.
- **Luokittelu:** ennustetaan kategorinen `performance_level` (tai rakennettu Low/Medium/High-luokka) ja pyritään parempaan tulokseen kuin enemmistöluokka.

Nämä ennusteet tukevat päätöksentekoa esimerkiksi opiskelijoiden tukitoimien kohdentamisessa.
        """
    )

    df_raw = load_data()
    df, reg_target, cls_target = choose_targets(df_raw)

    st.header("A) Data & EDA")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values (total)", int(df.isna().sum().sum()))

    st.subheader("Head")
    st.dataframe(df.head())

    st.subheader("Summary statistics")
    st.dataframe(df.describe(include="all").transpose())

    st.subheader("Simple distribution plot")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    hist_col = st.selectbox("Valitse histogrammin muuttuja", numeric_cols)
    fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
    ax_hist.hist(df[hist_col].dropna(), bins=25)
    ax_hist.set_title(f"Histogram: {hist_col}")
    ax_hist.set_xlabel(hist_col)
    ax_hist.set_ylabel("Count")
    st.pyplot(fig_hist)

    st.header("B) Modeling")

    st.subheader("Regression")
    st.write(
        "Target:",
        reg_target,
        "| Features:",
        "all other columns (with median/mode imputation + one-hot encoding).",
    )
    reg_results, reg_y_test, reg_predictions, cv_scores = evaluate_regression(df, reg_target)
    st.dataframe(reg_results.sort_values("RMSE"))
    st.caption(
        f"5-fold CV (Linear Regression, RMSE): {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
    )

    best_reg_row = reg_results.sort_values("RMSE").iloc[0]
    st.success(
        f"Paras regressiomalli: {best_reg_row['Model']} (RMSE={best_reg_row['RMSE']:.3f}, MAE={best_reg_row['MAE']:.3f})"
    )

    st.subheader("Classification")
    st.write(
        "Target:",
        cls_target,
        "| Features:",
        "all other columns (with median/mode imputation + one-hot encoding).",
    )
    cls_results, cls_y_test, confusion_data = evaluate_classification(df, cls_target)
    st.dataframe(cls_results.sort_values("F1", ascending=False))

    class_distribution = df[cls_target].value_counts(normalize=True).rename("ratio").reset_index()
    st.write("Class distribution:")
    st.dataframe(class_distribution)

    best_cls_row = cls_results.sort_values("F1", ascending=False).iloc[0]
    st.success(
        f"Paras luokittelumalli: {best_cls_row['Model']} (Accuracy={best_cls_row['Accuracy']:.3f}, F1={best_cls_row['F1']:.3f})"
    )

    st.header("C) Visualizations")
    reg_plot_model = st.selectbox("Regressiokuvan malli", list(reg_predictions.keys()), index=1)
    st.pyplot(regression_plot(reg_y_test, reg_predictions[reg_plot_model], reg_plot_model))
    st.caption("Predicted vs Actual: mitä lähempänä 45° viivaa pisteet ovat, sitä parempi malli.")

    cm_model = st.selectbox("Confusion matrix -malli", list(confusion_data.keys()), index=1)
    st.pyplot(confusion_plot(confusion_data[cm_model], labels=np.unique(cls_y_test), model_name=cm_model))
    st.caption("Confusion matrix näyttää, missä luokissa virheitä syntyy eniten.")

    st.header("D) Documentation")
    st.markdown(
        """
- **Supervised learning:** mallinnetaan yhteys syötteiden (features) ja targetin välillä tunnetulla opetusdatalla.
- **Regressio vs luokittelu tässä datassa:** regressio ennustaa pistemäärää (`exam_score`), luokittelu ennustaa suoriutumisluokkaa.
- **Mallit:** baseline + lineaarinen/logistinen malli + random forest (vertailu yksinkertaisen ja epälineaarisen mallin välillä).
- **Johtopäätös:** käytännössä hyödyllinen malli on sellainen, joka voittaa baseline-mallin selvästi ja pysyy vakaana cross-validationissa.
        """
    )


if __name__ == "__main__":
    main()
