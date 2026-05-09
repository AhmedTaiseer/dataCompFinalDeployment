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


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Amazon Return Predictor",
    layout="wide"
)

st.title("Amazon Product Return Prediction")
st.write("Optimized PCA + SVM Streamlit ML App")


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("amazon_sample_15k.csv")


df = load_data()
st.success(f"Dataset loaded ({len(df)} rows)")


# -------------------------------------------------
# FEATURE SETUP
# -------------------------------------------------
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


# -------------------------------------------------
# TRAINING PIPELINE (RUNS ONCE ONLY)
# -------------------------------------------------
@st.cache_resource
def train_model(data):

    X = data.drop('is_returned', axis=1)
    y = data['is_returned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    df_train = X_train.copy()
    df_train['is_returned'] = y_train

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

    # Encoding
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

    # PCA
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

    # Class weights
    classes = np.unique(y_train_final)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train_final
    )
    class_weight_dict = dict(zip(classes, weights))

    # MODEL (IMPORTANT FIX)
    svm_model = SVC(
        kernel='rbf',
        class_weight=class_weight_dict,
        probability=True,
        random_state=42
    )

    svm_model.fit(X_train_svm, y_train_final)
    y_pred = svm_model.predict(X_test_svm)

    return (
        svm_model,
        scaler,
        pca,
        categorical_dummy_cols,
        X_train_final,
        y_test_final,
        y_pred
    )


# -------------------------------------------------
# TRAIN MODEL ONCE
# -------------------------------------------------
with st.spinner("Training model (only once)..."):
    (
        svm_model,
        scaler,
        pca,
        categorical_dummy_cols,
        X_train_final,
        y_test_final,
        y_pred
    ) = train_model(df)

st.success("Model ready!")


# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
st.header("Model Evaluation")

accuracy = accuracy_score(y_test_final, y_pred)
precision = precision_score(y_test_final, y_pred, zero_division=0)
recall = recall_score(y_test_final, y_pred, zero_division=0)
f1 = f1_score(y_test_final, y_pred, zero_division=0)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy:.4f}")
c2.metric("Precision", f"{precision:.4f}")
c3.metric("Recall", f"{recall:.4f}")
c4.metric("F1 Score", f"{f1:.4f}")


# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test_final, y_pred)

fig, ax = plt.subplots(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Returned', 'Returned'],
    yticklabels=['Not Returned', 'Returned'],
    ax=ax
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

st.pyplot(fig)


# -------------------------------------------------
# CLASSIFICATION REPORT
# -------------------------------------------------
st.subheader("Classification Report")

report = classification_report(
    y_test_final,
    y_pred,
    target_names=['Not Returned', 'Returned'],
    zero_division=0
)

st.text(report)


# -------------------------------------------------
# USER INPUT PREDICTION (FAST)
# -------------------------------------------------
st.header("Predict Product Return")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Price", 0.0, 10000.0, 100.0)
    discount = st.number_input("Discount", 0.0, 100.0, 10.0)
    final_price = st.number_input("Final Price", 0.0, 10000.0, 90.0)
    rating = st.slider("Rating", 1.0, 5.0, 4.0)
    review_count = st.number_input("Review Count", 0, 100000, 100)

with col2:
    stock = st.number_input("Stock", 0, 10000, 50)
    seller_rating = st.slider("Seller Rating", 1.0, 5.0, 4.5)
    shipping_time_days = st.number_input("Shipping Days", 1, 30, 3)

    category = st.selectbox("Category", sorted(df['category'].dropna().unique()))
    brand = st.selectbox("Brand", sorted(df['brand'].dropna().unique()))
    payment_method = st.selectbox("Payment Method", sorted(df['payment_method'].dropna().unique()))


if st.button("Predict Return Probability"):

    input_df = pd.DataFrame({
        'price': [price],
        'discount': [discount],
        'final_price': [final_price],
        'rating': [rating],
        'review_count': [review_count],
        'stock': [stock],
        'seller_rating': [seller_rating],
        'shipping_time_days': [shipping_time_days],
        'category': [category],
        'brand': [brand],
        'payment_method': [payment_method]
    })

    # Encoding
    input_encoded = pd.get_dummies(
        input_df,
        columns=categorical_cols,
        drop_first=True
    )

    input_encoded = input_encoded.reindex(
        columns=X_train_final.columns,
        fill_value=0
    )

    input_num = input_encoded[numerical_cols]

    input_scaled = scaler.transform(input_num)
    input_pca = pca.transform(input_scaled)

    input_cat = input_encoded[categorical_dummy_cols].values

    input_svm = np.hstack([input_pca, input_cat])

    prediction = svm_model.predict(input_svm)[0]

    # FAST probability (no heavy calibration)
    decision = svm_model.decision_function(input_svm)[0]
    probability = 1 / (1 + np.exp(-decision))

    st.subheader("Result")

    if prediction == 1:
        st.error("LIKELY RETURNED")
    else:
        st.success("LIKELY NOT RETURNED")

    st.metric("Return Probability", f"{probability * 100:.2f}%")
