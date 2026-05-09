import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

st.set_page_config(page_title="Amazon Return Predictor", layout="wide")

st.title("Amazon Product Return Prediction")
st.write("Upload your Amazon ecommerce CSV dataset.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    # Load dataset
    df_full = pd.read_csv("amazon_sample_15k.csv")
    df = df_full.sample(n=min(15000, len(df_full)), random_state=42)

    st.success(f"Loaded {len(df)} rows")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Split
    X = df.drop('is_returned', axis=1)
    y = df['is_returned']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    df_train = X_train.copy()
    df_train['is_returned'] = y_train

    numerical_cols = [
        'price',
        'discount',
        'final_price',
        'rating',
        'review_count',
        'stock',
        'seller_rating',
        'shipping_time_days'
    ]

    categorical_cols = [
        'category',
        'brand',
        'payment_method'
    ]

    # --------------------
    # EDA SECTION
    # --------------------
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(
            data=df_train,
            x='rating',
            order=sorted(df_train['rating'].dropna().unique()),
            ax=ax
        )
        plt.xticks(rotation=45)
        plt.title("Rating Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        category_counts = df_train['category'].value_counts().head(10)
        sns.barplot(
            x=category_counts.values,
            y=category_counts.index,
            ax=ax
        )
        plt.title("Top Categories")
        st.pyplot(fig)

    # --------------------
    # CLEANING
    # --------------------
    keep_cols = numerical_cols + categorical_cols + ['is_returned']

    df_train_clean = df_train[keep_cols].copy()
    X_test_clean = X_test[keep_cols[:-1]].copy()
    X_test_clean['is_returned'] = y_test

    # Outlier clipping
    for col in numerical_cols:
        Q1 = df_train_clean[col].quantile(0.25)
        Q3 = df_train_clean[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_train_clean[col] = df_train_clean[col].clip(lower, upper)
        X_test_clean[col] = X_test_clean[col].clip(lower, upper)

    # One-hot encoding
    df_train_encoded = pd.get_dummies(
        df_train_clean,
        columns=categorical_cols,
        drop_first=True
    )

    X_test_encoded = pd.get_dummies(
        X_test_clean,
        columns=categorical_cols,
        drop_first=True
    )

    X_test_encoded = X_test_encoded.reindex(
        columns=df_train_encoded.columns,
        fill_value=0
    )

    X_train_final = df_train_encoded.drop('is_returned', axis=1)
    y_train_final = df_train_encoded['is_returned']

    X_test_final = X_test_encoded.drop('is_returned', axis=1)
    y_test_final = X_test_encoded['is_returned']

    # --------------------
    # PCA
    # --------------------
    X_train_num = X_train_final[numerical_cols]
    X_test_num = X_test_final[numerical_cols]

    categorical_dummy_cols = [
        c for c in X_train_final.columns
        if c not in numerical_cols
    ]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    pca = PCA(n_components=0.95)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    X_train_svm = np.hstack([
        X_train_pca,
        X_train_final[categorical_dummy_cols].values
    ])

    X_test_svm = np.hstack([
        X_test_pca,
        X_test_final[categorical_dummy_cols].values
    ])

    # --------------------
    # TRAIN MODEL
    # --------------------
    st.header("Training SVM Model")

    classes = np.unique(y_train_final)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train_final
    )

    class_weight_dict = dict(zip(classes, weights))

    svm_model = SVC(
        kernel='rbf',
        class_weight=class_weight_dict,
        random_state=42
    )

    svm_model.fit(X_train_svm, y_train_final)

    y_pred = svm_model.predict(X_test_svm)

    # --------------------
    # METRICS
    # --------------------
    st.header("Model Evaluation")

    accuracy = accuracy_score(y_test_final, y_pred)
    precision = precision_score(y_test_final, y_pred)
    recall = recall_score(y_test_final, y_pred)
    f1 = f1_score(y_test_final, y_pred)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{accuracy:.4f}")
    c2.metric("Precision", f"{precision:.4f}")
    c3.metric("Recall", f"{recall:.4f}")
    c4.metric("F1 Score", f"{f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test_final, y_pred)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Returned', 'Returned'],
        yticklabels=['Not Returned', 'Returned'],
        ax=ax
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(
        y_test_final,
        y_pred,
        target_names=['Not Returned', 'Returned']
    )
    st.text(report)
