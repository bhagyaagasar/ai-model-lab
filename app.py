import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="AI Model Lab", page_icon="🤖", layout="wide")

st.title("🤖 AI Model Lab")
st.caption("Upload a tabular dataset, train a machine-learning classifier, evaluate it, and make predictions.")

with st.sidebar:
    st.header("1. Dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random state", 0, 999, 42)

if "model" not in st.session_state:
    st.session_state.model = None
if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = None
if "encoders" not in st.session_state:
    st.session_state.encoders = {}
if "target_encoder" not in st.session_state:
    st.session_state.target_encoder = None

if not uploaded:
    st.info("Upload a CSV file from the sidebar to begin.")
    st.markdown("""
    ### What this app does
    - Dataset inspection and preprocessing
    - Train/test split
    - Logistic Regression, Random Forest, SVM, or KNN
    - Accuracy, precision, recall and F1-score
    - Confusion matrix and classification report
    - Single-row prediction
    """)
    st.stop()

df = pd.read_csv(uploaded)
st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

tab1, tab2, tab3 = st.tabs(["📊 Explore", "🧠 Train & Evaluate", "🔮 Predict"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", f"{len(df.columns)-1:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Column information")
    info = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(info, use_container_width=True)

with tab2:
    target = st.selectbox("Target column", df.columns, index=len(df.columns)-1)
    model_name = st.selectbox(
        "Model",
        ["Random Forest", "Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors"]
    )

    if st.button("🚀 Train model", type="primary"):
        work = df.dropna(subset=[target]).copy()
        X = work.drop(columns=[target])
        y = work[target].astype(str)

        # Encode categorical feature columns
        encoders = {}
        for col in X.columns:
            if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna("__MISSING__").astype(str))
                encoders[col] = le
        # Convert any remaining values to numeric
        X = X.apply(pd.to_numeric, errors="coerce")

        target_encoder = LabelEncoder()
        y_enc = target_encoder.fit_transform(y)

        stratify = y_enc if len(np.unique(y_enc)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, random_state=int(random_state), stratify=stratify
        )

        if model_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=200, random_state=int(random_state), n_jobs=-1)
        elif model_name == "Logistic Regression":
            clf = LogisticRegression(max_iter=2000)
        elif model_name == "Support Vector Machine":
            clf = SVC(kernel="rbf", probability=True)
        else:
            clf = KNeighborsClassifier(n_neighbors=min(5, max(1, len(X_train)-1)))

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        metrics = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_test, pred, average="weighted", zero_division=0),
        }

        st.session_state.model = pipe
        st.session_state.feature_columns = list(X.columns)
        st.session_state.encoders = encoders
        st.session_state.target_encoder = target_encoder

        st.subheader("Evaluation metrics")
        cols = st.columns(4)
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.2%}")

        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(cm)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(len(target_encoder.classes_)))
        ax.set_yticks(range(len(target_encoder.classes_)))
        ax.set_xticklabels(target_encoder.classes_, rotation=45, ha="right")
        ax.set_yticklabels(target_encoder.classes_)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.tight_layout()
        st.pyplot(fig)

        report = classification_report(
            y_test, pred, target_names=target_encoder.classes_, zero_division=0, output_dict=True
        )
        st.subheader("Classification report")
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

with tab3:
    if st.session_state.model is None:
        st.warning("Train a model first.")
    else:
        st.subheader("Single-row prediction")
        values = {}
        for col in st.session_state.feature_columns:
            values[col] = st.text_input(f"{col}", value="")
        if st.button("🔮 Predict"):
            row = pd.DataFrame([values])
            for col, le in st.session_state.encoders.items():
                try:
                    row[col] = le.transform(row[col].fillna("__MISSING__").astype(str))
                except ValueError:
                    st.error(f"Unknown category supplied for '{col}'.")
                    st.stop()
            row = row.apply(pd.to_numeric, errors="coerce")
            pred = st.session_state.model.predict(row)[0]
            label = st.session_state.target_encoder.inverse_transform([pred])[0]
            st.success(f"Prediction: **{label}**")
            if hasattr(st.session_state.model, "predict_proba"):
                probs = st.session_state.model.predict_proba(row)[0]
                prob_df = pd.DataFrame({
                    "Class": st.session_state.target_encoder.classes_,
                    "Probability": probs
                }).sort_values("Probability", ascending=False)
                st.dataframe(prob_df, hide_index=True, use_container_width=True)
