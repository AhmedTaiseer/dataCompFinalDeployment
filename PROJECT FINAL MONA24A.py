# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load dataset and take 15000 random entries
df_full = pd.read_csv(r"C:\Users\ahmed\Desktop\data computation\amazon_ecommerce_1M.csv")
df = df_full.sample(n=15000, random_state=42)
print(f"Using {len(df)} random entries")

# Check class distribution
print("\nClass distribution (is_returned):")
print(df['is_returned'].value_counts())
print(f"Returned percentage: {df['is_returned'].mean() * 100:.2f}%")

# TRAIN TEST SPLIT (BEFORE ANYTHING ELSE)
X = df.drop('is_returned', axis=1)
y = df['is_returned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# EDA (BEFORE CLEANING)
df_train = X_train.copy()
df_train['is_returned'] = y_train

# Plot 1: Distribution of ratings
plt.figure(figsize=(10, 6))
sns.countplot(data=df_train, x='rating', order=sorted(df_train['rating'].dropna().unique()))
plt.title('Distribution of Product Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Top 10 product categories
plt.figure(figsize=(12, 6))
category_counts = df_train['category'].value_counts().head(10)
sns.barplot(x=category_counts.values, y=category_counts.index, hue=category_counts.index, legend=False, palette='viridis')
plt.title('Top 10 Product Categories by Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Category')
plt.tight_layout()
plt.show()

# Plot 3: Relationship between final price and rating
plt.figure(figsize=(10, 6))
filtered_df = df_train[df_train['final_price'] < df_train['final_price'].quantile(0.99)]
sns.boxplot(data=filtered_df, x='rating', y='final_price')
plt.title('Final Price Distribution by Rating')
plt.xlabel('Rating')
plt.ylabel('Final Price')
plt.yscale('log')
plt.tight_layout()
plt.show()

# Plot 4: Correlation heatmap
plt.figure(figsize=(10, 8))
numerical_cols = ['price', 'discount', 'final_price', 'rating', 'review_count', 'stock', 'seller_rating', 'shipping_time_days']
corr_matrix = df_train[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()

# Plot 5: Boxplot for anomaly detection
plt.figure(figsize=(12, 8))
df_train[numerical_cols].boxplot()
plt.title('Boxplot of Numerical Features - Outlier Detection')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# DATA CLEANING

# Keep numerical columns + the 3 categorical columns we want + target
categorical_cols_to_keep = ['category', 'brand', 'payment_method']
keep_cols = numerical_cols + categorical_cols_to_keep + ['is_returned']

df_train_clean = df_train[keep_cols].copy()
X_test_clean = X_test[keep_cols[:-1]].copy()
X_test_clean['is_returned'] = y_test

print(f"\nTraining set shape after cleaning: {df_train_clean.shape}")
print(f"Test set shape after cleaning: {X_test_clean.shape}")

# Handle outliers in numerical columns (cap instead of drop)
for col in numerical_cols:
    Q1 = df_train_clean[col].quantile(0.25)
    Q3 = df_train_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_train_clean[col] = df_train_clean[col].clip(lower_bound, upper_bound)
    X_test_clean[col] = X_test_clean[col].clip(lower_bound, upper_bound)

# ENCODE CATEGORICAL COLUMNS (one-hot encoding)
df_train_encoded = pd.get_dummies(df_train_clean, columns=categorical_cols_to_keep, drop_first=True)
X_test_encoded = pd.get_dummies(X_test_clean, columns=categorical_cols_to_keep, drop_first=True)

# Align columns between train and test
X_test_encoded = X_test_encoded.reindex(columns=df_train_encoded.columns, fill_value=0)

# Separate features and target
X_train_final = df_train_encoded.drop('is_returned', axis=1)
y_train_final = df_train_encoded['is_returned']
X_test_final = X_test_encoded.drop('is_returned', axis=1)
y_test_final = X_test_encoded['is_returned']

print(f"Final features shape - Train: {X_train_final.shape}, Test: {X_test_final.shape}")

# DIMENSIONALITY REDUCTION (PCA)
# Only apply PCA to numerical columns, keep categorical dummies as-is
numerical_cols_pca = numerical_cols
X_train_numerical = X_train_final[numerical_cols_pca]
X_test_numerical = X_test_final[numerical_cols_pca]

# Get categorical dummy columns
categorical_dummy_cols = [col for col in X_train_final.columns if col not in numerical_cols_pca]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numerical)
X_test_scaled = scaler.transform(X_test_numerical)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Numerical features: {len(numerical_cols_pca)} PCA reduced to: {X_train_pca.shape[1]}")
print(f"Categorical dummy features: {len(categorical_dummy_cols)}")
print(f"Total features for SVM: {X_train_pca.shape[1] + len(categorical_dummy_cols)}")

# Combine PCA components with categorical dummies
X_train_svm = np.hstack([X_train_pca, X_train_final[categorical_dummy_cols].values])
X_test_svm = np.hstack([X_test_pca, X_test_final[categorical_dummy_cols].values])

print(f"SVM input shape - Train: {X_train_svm.shape}, Test: {X_test_svm.shape}")

# SVM MODEL WITH CLASS WEIGHT
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train_final)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_final)
class_weight_dict = dict(zip(classes, class_weights))
print(f"\nClass weights: {class_weight_dict}")

svm_model = SVC(kernel='rbf', class_weight=class_weight_dict, random_state=42)
svm_model.fit(X_train_svm, y_train_final)

y_pred = svm_model.predict(X_test_svm)

# EVALUATION
print("SVM MODEL EVALUATION")
print(f"Accuracy: {accuracy_score(y_test_final, y_pred):.4f}")
print(f"Precision: {precision_score(y_test_final, y_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_final, y_pred, zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test_final, y_pred, zero_division=0):.4f}")

print("\nClassification Report:")
print(classification_report(y_test_final, y_pred, target_names=['Not Returned', 'Returned'], zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test_final, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Returned', 'Returned'], yticklabels=['Not Returned', 'Returned'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()