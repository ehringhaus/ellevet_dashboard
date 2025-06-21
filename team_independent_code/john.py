#!/usr/bin/env python
# coding: utf-8

# In[32]:


# ElleVet Churn Model – Final Version

# Author: John Lane  
# Model Type: Logistic Regression with Custom Feature Engineering  
# Purpose: Identify churn risk among one-time buyers using behavioral signals  
# Deliverables: Trained model, risk scores, visualizations, and top 10% export

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, roc_curve, classification_report
)

# --- Step 1: Load & Prepare Data ---
base_path = "/Users/johnlane/Desktop/CAPSTONE FINAL DATA"

customers     = pd.read_csv(os.path.join(base_path, "customers_redacted.csv"))
orders        = pd.read_csv(os.path.join(base_path, "orders_redacted.csv"), low_memory=False)
refunds       = pd.read_csv(os.path.join(base_path, "refunds_affiliated.csv"))
tickets       = pd.read_csv(os.path.join(base_path, "tickets_redacted.csv"))
subscriptions = pd.read_csv(os.path.join(base_path, "subscriptions_redacted.csv"))

order_counts = orders.groupby("customer_id")["order_id"].nunique().reset_index(name="order_count")
customers = customers.merge(order_counts, on="customer_id", how="left")
customers["order_count"] = customers["order_count"].fillna(0).astype(int)
customers["returned"] = (customers["order_count"] > 1).astype(int)

# --- Step 2: Feature Engineering ---
df = customers.copy()
orders["order_id"] = orders["order_id"].astype(str)
orders["order_total"] = pd.to_numeric(orders["order_total"], errors="coerce")

refunds["order_id"] = refunds["order_id"].astype(str)
if "customer_id" in refunds.columns:
    refunds = refunds.drop(columns=["customer_id"])

refunds_merged = refunds.merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
refunds_merged = refunds_merged.dropna(subset=["customer_id"])
refunds_merged["customer_id"] = refunds_merged["customer_id"].astype(int)
refund_flag = refunds_merged[["customer_id"]].drop_duplicates()
refund_flag["refund_flag"] = 1

df = df.merge(refund_flag, on="customer_id", how="left", indicator=True)
df["refund_flag"] = df["refund_flag"].fillna(0).astype(int)
df.drop(columns=[col for col in df.columns if col.startswith("_merge")], inplace=True)

customer_spend = orders.groupby("customer_id")["order_total"].sum().reset_index().rename(columns={"order_total": "total_spend_temp"})
df = df.merge(customer_spend, on="customer_id", how="left", suffixes=('', '_dup'))
df["total_spend"] = df["total_spend_temp"].fillna(0)
df.drop(columns=["total_spend_temp"], inplace=True)

df["spend_bin"] = pd.qcut(df["total_spend"], 3, labels=["Low", "Mid", "High"])
df = pd.get_dummies(df, columns=["spend_bin"], prefix="spend")

for col in ["spend_Low", "spend_Mid", "spend_High"]:
    if col not in df.columns:
        df[col] = 0
df[["spend_Low", "spend_Mid", "spend_High"]] = df[["spend_Low", "spend_Mid", "spend_High"]].astype(int)

df["slow_response_flag"] = 0

orders["discount_used"] = orders["cart_discount"].fillna(0).astype(float) > 0
discount_used_flag = orders.groupby("customer_id")["discount_used"].max().reset_index()
df = df.merge(discount_used_flag, on="customer_id", how="left")
df["discount_used"] = df["discount_used"].fillna(0).astype(int)

tickets["email_hash"] = tickets["requester_email"].str.lower().str.strip()
tickets["tags"] = tickets["tags"].fillna("").str.lower()
tag_merge = tickets.groupby("email_hash")["tags"].apply(lambda x: ",".join(x)).reset_index()
df = df.merge(tag_merge, on="email_hash", how="left")
df["tags"] = df["tags"].fillna("")

df["escalated_flag"] = df["tags"].str.contains("escalated").astype(int)
df["site_flag"] = df["tags"].str.contains("site|website|checkout|page").astype(int)

df["refund_slow_interaction"] = ((df["refund_flag"] == 1) & (df["slow_response_flag"] == 1)).astype(int)
df["site_refund_interaction"] = ((df["refund_flag"] == 1) & (df["site_flag"] == 1)).astype(int)
df["discount_refund_interaction"] = ((df["refund_flag"] == 1) & (df["discount_used"] == 1)).astype(int)

feature_cols = [
    "refund_flag", "slow_response_flag", "discount_used", "escalated_flag", "site_flag",
    "refund_slow_interaction", "site_refund_interaction", "discount_refund_interaction",
    "spend_Low", "spend_Mid"
]

# --- Step 3: Model Training ---
X = df[feature_cols].copy()
y = df["returned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("✅ Logistic Regression Metrics:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_proba):.3f}")

# --- Step 4: Visualizations ---

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Coefficient Plot
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_[0]}).sort_values("Coefficient")
plt.figure(figsize=(6, 4))
sns.barplot(x="Coefficient", y="Feature", data=coef_df, hue="Feature", palette="coolwarm", legend=False)
plt.title("Logistic Regression Coefficients")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Risk Score Distribution
risk_df = X_test.copy()
risk_df["Churn_Probability"] = y_proba
plt.figure(figsize=(6, 4))
sns.histplot(risk_df["Churn_Probability"], bins=25, kde=True, color="teal")
plt.title("Churn Risk Score Distribution")
plt.xlabel("Predicted Probability of Return")
plt.ylabel("Customer Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Return Rate by Spend Tier
tier_df = df.copy()
tier_df["spend_tier"] = pd.qcut(tier_df["total_spend"], 3, labels=["Low", "Mid", "High"])
return_rates = (
    tier_df.groupby("spend_tier", observed=True)["returned"]
           .mean()
           .reset_index()
           .rename(columns={"returned": "Return Rate"})
           .sort_values("Return Rate", ascending=False)
)

plt.figure(figsize=(6, 4))
sns.barplot(
    data=return_rates, 
    x="spend_tier", 
    y="Return Rate", 
    hue="spend_tier", 
    palette="Blues_r", 
    legend=False
)
plt.title("Return Rate by Spend Tier")
plt.xlabel("Spend Tier")
plt.ylabel("Return Rate")
plt.ylim(0, 1)
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

# 5. Churn Behaviors by Risk Tier
risk_df["Risk_Tier"] = pd.qcut(
    risk_df["Churn_Probability"], q=[0, 0.33, 0.66, 1.0],
    labels=["Low", "Mid", "High"]
)

focus_features = ["discount_used", "refund_flag", "discount_refund_interaction"]
for col in focus_features:
    if col not in risk_df.columns:
        risk_df[col] = df[col]

risk_summary = (
    risk_df.groupby("Risk_Tier", observed=True)[focus_features]
           .mean()
           .T
           .reset_index()
           .rename(columns={"index": "Feature"})
)
risk_long = risk_summary.melt(id_vars="Feature", var_name="Risk_Tier", value_name="Proportion")
tier_colors = {"Low": "green", "Mid": "gold", "High": "red"}

plt.figure(figsize=(9, 6))
barplot = sns.barplot(
    data=risk_long,
    x="Feature", y="Proportion", hue="Risk_Tier",
    palette=tier_colors
)
for p in barplot.patches:
    height = p.get_height()
    if height > 0.01:
        barplot.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=8)
plt.title("Top Churn-Related Behaviors by Risk Tier")
plt.ylabel("Proportion of Customers")
plt.xlabel("Feature")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y")
plt.legend(title="Risk Tier")
plt.tight_layout()
plt.show()

# --- Step 6: Export Top 10% High-Risk Customers ---
X_all = df[feature_cols].copy()
df["Predicted_Prob"] = model.predict_proba(X_all)[:, 1]
cutoff = df["Predicted_Prob"].quantile(0.90)
top_10_risk = df[df["Predicted_Prob"] >= cutoff].copy()
top_10_risk = top_10_risk.sort_values("Predicted_Prob", ascending=False)

cols_to_export = [
    "customer_id", "Predicted_Prob", "refund_flag", "discount_used",
    "discount_refund_interaction", "total_spend", "returned"
]
top_10_risk[cols_to_export].to_csv("top_10_percent_high_risk_customers.csv", index=False)
print(f"✅ Exported {len(top_10_risk)} high-risk customers to CSV ➜ top_10_percent_high_risk_customers.csv")

# Print top 5 for personal review
print("🔎 Top 5 High-Risk Customers Preview:")
print(top_10_risk[cols_to_export].head())


# In[ ]:




