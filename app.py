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
st.write("Machine Learning model using PCA + SVM")


# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("amazon_sample_15k.csv")
    return df

with st.spinner("Loading dataset..."):
    df = load_data()

st.success(f"Dataset loaded successfully! ({len(df)} rows)")


# -------------------------------------------------
# DATASET PREVIEW
# -------------------------------------------------
st.header("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

with col2:
    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Return Distribution")
    st.write(df['is_returned'].value_counts())

    returned_percent = df['is_returned'].mean() * 100
    st.write(f"Returned Percentage: {returned_percent:.2f}%")


# -------------------------------------------------
# TRAIN TEST SPLIT
# -------------------------------------------------
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


# -------------------------------------------------
# FEATURE COLUMNS
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
# EDA SECTION
# -------------------------------------------------
st.header("Exploratory Data Analysis")

# Rating distribution
fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(
    data=df_train,
    x='rating',
    order=sorted(df_train['rating'].dropna().unique()),
    ax=ax
)

plt.title("Distribution of Product Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.xticks(rotation=45)

st.pyplot(fig)


# Top categories
fig, ax = plt.subplots(figsize=(10, 5))

category_counts = df_train['category'].value_counts().head(10)

sns.barplot(
    x=category_counts.values,
    y=category_counts.index,
    ax=ax
)

plt.title("Top 10 Product Categories")
plt.xlabel("Count")
plt.ylabel("Category")

st.pyplot(fig)


# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 7))

corr_matrix = df_train[numerical_cols].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    ax=ax
)

plt.title("Correlation Heatmap")

st.pyplot(fig)


# -------------------------------------------------
# DATA CLEANING
# -------------------------------------------------
keep_cols = numerical_cols + categorical_cols + ['is_returned']

df_train_clean = df_train[keep_cols].copy()

X_test_clean = X_test[keep_cols[:-1]].copy()
X_test_clean['is_returned'] = y_test


# Handle outliers
for col in numerical_cols:

    Q1 = df_train_clean[col].quantile(0.25)
    Q3 = df_train_clean[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_train_clean[col] = df_train_clean[col].clip(lower, upper)
    X_test_clean[col] = X_test_clean[col].clip(lower, upper)


# -------------------------------------------------
# ENCODING
# -------------------------------------------------
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


# Separate features and target
X_train_final = df_train_encoded.drop('is_returned', axis=1)
y_train_final = df_train_encoded['is_returned']

X_test_final = X_test_encoded.drop('is_returned', axis=1)
y_test_final = X_test_encoded['is_returned']


# -------------------------------------------------
# PCA
# -------------------------------------------------
st.header("PCA Dimensionality Reduction")

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

st.write(f"Original numerical features: {len(numerical_cols)}")
st.write(f"PCA reduced dimensions: {X_train_pca.shape[1]}")

# Combine PCA with categorical dummies
X_train_svm = np.hstack([
    X_train_pca,
    X_train_final[categorical_dummy_cols].values
])

X_test_svm = np.hstack([
    X_test_pca,
    X_test_final[categorical_dummy_cols].values
])


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
st.header("Training SVM Model")

with st.spinner("Training SVM model..."):

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

st.success("Model training completed!")


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
