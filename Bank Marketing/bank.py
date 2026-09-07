import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "bank-full.csv")

IMG_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, name), bbox_inches="tight")
    plt.close()
    print(f"saved -> {IMG_DIR}/{name}")


# 1. LOAD DATA

df = pd.read_csv(DATA_PATH, sep=";")
print("Shape:", df.shape)

num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
cat_cols = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "poutcome"]
target = "y"

with open(os.path.join(BASE_DIR, "eda_summary.txt"), "w") as f:
    f.write("BANK MARKETING DATASET - EDA SUMMARY\n" + "=" * 50 + "\n\n")
    f.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")
    f.write("Dtypes:\n" + str(df.dtypes) + "\n\n")
    f.write("Missing values:\n" + str(df.isnull().sum()) + "\n\n")
    f.write("Describe (numeric):\n" + str(df[num_cols].describe()) + "\n\n")
    f.write("Target distribution:\n" + str(df[target].value_counts()) + "\n")


# 2. EDA PLOTS 

plt.figure(figsize=(5, 4))
ax = sns.countplot(x=target, data=df, palette=["#3498db", "#e67e22"])
plt.title("Target Distribution: Term Deposit Subscription (y)")
for p in ax.patches:
    pct = 100 * p.get_height() / len(df)
    ax.annotate(f"{p.get_height()} ({pct:.1f}%)", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom")
savefig("01_target_distribution.png")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color="#2980b9")
    axes[i].set_title(f"Distribution of {col}")
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])
savefig("02_numerical_distributions.png")

plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap - Numerical Features")
savefig("03_correlation_heatmap.png")

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    rate = df.groupby(col)[target].apply(lambda s: (s == "yes").mean() * 100).sort_values()
    rate.plot(kind="barh", ax=axes[i], color="#27ae60")
    axes[i].set_title(f"Subscription Rate (%) by {col}")
for j in range(len(cat_cols), len(axes)):
    fig.delaxes(axes[j])
savefig("04_categorical_vs_target_rate.png")


# 3. PREPROCESSING

model_df = df.copy()

# Drop 'duration' to avoid data leakage (only known AFTER a call happens,
# so it can't be used for genuine pre-call prediction).
model_df = model_df.drop(columns=["duration"])
num_cols_model = [c for c in num_cols if c != "duration"]

# One-hot encode categoricals
model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)

# Encode target
model_df[target] = (model_df[target] == "yes").astype(int)

X = model_df.drop(columns=[target])
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric columns only (dummies stay 0/1)
scaler = StandardScaler()
X_train[num_cols_model] = scaler.fit_transform(X_train[num_cols_model])
X_test[num_cols_model] = scaler.transform(X_test[num_cols_model])


# 4. MODEL: LOGISTIC REGRESSION (class_weight='balanced' for imbalance)

model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, target_names=["no", "yes"])
cm = confusion_matrix(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)

print(report)
print("Confusion matrix:\n", cm)
print("ROC-AUC:", auc)

with open(os.path.join(BASE_DIR, "model_report.txt"), "w") as f:
    f.write("LOGISTIC REGRESSION - MODEL EVALUATION\n" + "=" * 50 + "\n\n")
    f.write(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")
    f.write(f"Features used: {X.shape[1]} (duration excluded to prevent leakage)\n\n")
    f.write("Classification report:\n" + report + "\n")
    f.write("Confusion matrix:\n" + str(cm) + "\n\n")
    f.write(f"ROC-AUC: {auc:.4f}\n")
    f.write(f"F1 score (yes class): {f1:.4f}\n")


# 5. EVALUATION PLOTS

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["no", "yes"], yticklabels=["no", "yes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
savefig("05_confusion_matrix.png")

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5.5, 5))
plt.plot(fpr, tpr, color="#e67e22", label=f"ROC curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
savefig("06_roc_curve.png")

prec, rec, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(5.5, 5))
plt.plot(rec, prec, color="#8e44ad")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
savefig("07_precision_recall_curve.png")

# Feature importance via coefficients
coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
top_features = pd.concat([coef.head(10), coef.tail(10)])
plt.figure(figsize=(8, 8))
top_features.plot(kind="barh", color=["#c0392b" if v < 0 else "#27ae60" for v in top_features])
plt.title("Top 20 Logistic Regression Coefficients")
plt.xlabel("Coefficient (standardized features)")
savefig("08_feature_importance.png")

print("\nDone. See model_report.txt and images/ for full results.")
