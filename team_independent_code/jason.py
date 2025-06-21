# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:53:26 2025

@author: Jason.Hickson
"""


#  HIGH-RISK SCORING FOR ONE-TIME NON-SUBSCRIBERS

import pandas as pd, numpy as np, hashlib, ast, re
import matplotlib.pyplot as plt, seaborn as sns
from datetime import datetime

# LOAD DATA 
base = r"C:/Users/Jason.Hickson/OneDrive - State of Maine/Documents/Jason/ALY 6980/project data"

orders        = pd.read_csv(fr"{base}/orders_redacted.csv")
subscriptions = pd.read_csv(fr"{base}/subscriptions_redacted.csv")
tickets       = pd.read_csv(fr"{base}/tickets_redacted.csv")
refunds       = pd.read_csv(fr"{base}/refunds_affiliated.csv")
customers     = pd.read_csv(fr"{base}/customers_redacted.csv")
orders_utm    = pd.read_csv(fr"{base}/orders_with_utm.csv")
quizzes       = pd.read_csv(fr"{base}/quizzes_redacted.csv")


# ONE-TIME  (non-subscriber) COHORT  
orders["customer_id"] = orders["customer_id"].astype(int)  
order_counts = (
    orders.groupby("customer_id", as_index=False)["order_id"]
          .nunique()
          .rename(columns={"order_id": "num_orders"})
)
one_time_non_subs = order_counts.query("num_orders == 1").copy()
sub_ids = subscriptions["subscription_customer_user_id"].unique()
one_time_non_subs = one_time_non_subs[~one_time_non_subs["customer_id"].isin(sub_ids)]

# add email_hash
one_time_non_subs = one_time_non_subs.merge(
    orders[["customer_id", "email_hash"]].dropna().drop_duplicates(),
    on="customer_id", how="left"
)

#ZENDESK TAG FLAGS & COUNTS 
def sha_email(e): 
    return np.nan if pd.isna(e) else hashlib.sha256(e.strip().lower().encode()).hexdigest()

tickets["email_hash"] = tickets["requester_email"].apply(sha_email)
tickets["tags"] = tickets["tags"].fillna("").str.lower()
tickets["tag_list"] = tickets["tags"].str.split(",")

tix_ex = tickets.explode("tag_list")
tix_ex["tag_list"] = tix_ex["tag_list"].str.strip()

# binary tag flags
tix_ex["has_cancel_tag"]  = tix_ex["tag_list"].str.contains(r"cancel")
tix_ex["has_urgent_tag"]  = tix_ex["tag_list"].str.contains(r"urgent")
tix_ex["mentions_dosing"] = tix_ex["tag_list"].str.contains(r"dosing")
tix_ex["product_issue"]   = tix_ex["tag_list"].str.contains(r"product_|quality|broken|damage|leak")

ticket_flags = (
    tix_ex.groupby("email_hash")[["has_cancel_tag","has_urgent_tag",
                                  "mentions_dosing","product_issue"]]
          .max()
          .reset_index()
)
ticket_counts = (
    tix_ex.groupby("email_hash").size()
          .reset_index(name="num_tickets")
)
ticket_features = ticket_flags.merge(ticket_counts, on="email_hash", how="outer")

# merge into cohort
one_time_non_subs = one_time_non_subs.merge(ticket_features, on="email_hash", how="left")

flag_cols = ["has_cancel_tag","has_urgent_tag","mentions_dosing","product_issue","num_tickets"]
for col in flag_cols:                      # make sure column exists
    if col not in one_time_non_subs.columns:
        one_time_non_subs[col] = 0
# nan-safe cast to int
for col in flag_cols:
    one_time_non_subs[col] = (
        pd.to_numeric(one_time_non_subs[col], errors="coerce")
          .fillna(0)
          .astype(int)
    )

# REFUND FLAG 
refunds = refunds.merge(orders[["order_id","customer_id"]], on="order_id", how="left")
refund_flag = refunds[["customer_id"]].drop_duplicates()
refund_flag["refunded"] = 1

# FIRST-ORDER & PRODUCT FORM 
orders["completed_date"] = pd.to_datetime(orders["completed_date"], errors="coerce")

first_order = (
    orders.sort_values("completed_date")
          .drop_duplicates("customer_id")
          .loc[:, ["customer_id","order_total","completed_date"]]
          .rename(columns={"order_total":"first_order_total",
                           "completed_date":"first_order_date"})
)

# parse product form from line_items
def extract_names(x):
    try:
        items = ast.literal_eval(x) if isinstance(x,str) else []
        return [i.get("product_name","") for i in items if isinstance(i,dict)]
    except Exception:
        return []
def classify_form(n):
    if not isinstance(n,str): return "Other"
    s = n.lower()
    if "chew"     in s: return "Chew"
    if "softgel"  in s or "gel" in s: return "Softgel"
    if "oil"      in s: return "Oil"
    if "capsule"  in s: return "Capsule"
    return "Other"

orders["product_names"] = orders["line_items"].apply(extract_names)
exp = orders.explode("product_names")
exp["product_form"] = exp["product_names"].apply(classify_form)
first_form = exp.sort_values("completed_date").drop_duplicates("customer_id")[["customer_id","product_form"]]

# UTM RISK FLAG 
orders_with_utm = orders[["order_id","customer_id"]].merge(
    orders_utm[["order_id","utm_source"]], on="order_id", how="left"
)
first_utm = orders_with_utm.dropna(subset=["utm_source"]).drop_duplicates("customer_id")
first_utm["utm_source_clean"] = first_utm["utm_source"].str.lower()

bad_words = ["facebook","fb","affiliate","coupon","discount"]
first_utm["utm_risk_flag"] = first_utm["utm_source_clean"].apply(
    lambda x: int(any(w in x for w in bad_words))
)


# QUIZ FLAGS 
quiz = quizzes.rename(columns={"email":"email_hash"})
quiz_flags = quiz[["email_hash","16._pet_type","24._situational_stress"]].copy()

quiz_flags["cat_flag"] = (
    quiz_flags["16._pet_type"].fillna("")
               .str.lower()
               .str.contains("cat")
               .astype(int)
)

quiz_flags["stress_flag"] = (
    quiz_flags["24._situational_stress"].fillna("")
               .str.lower()
               .str.contains("stress")
               .astype(int)
)

quiz_flags = quiz_flags[["email_hash","cat_flag","stress_flag"]]

# MASTER FRAME 
df = (one_time_non_subs
      .merge(first_order,  on="customer_id", how="left")
      .merge(first_form,   on="customer_id", how="left")
      .merge(first_utm[["customer_id","utm_risk_flag"]], on="customer_id", how="left")
      .merge(quiz_flags,   on="email_hash", how="left")
      .merge(refund_flag,  on="customer_id", how="left")
)

for col in ["utm_risk_flag","cat_flag","stress_flag","refunded"]:
    df[col] = df[col].fillna(0).astype(int)

df["first_order_total"] = df["first_order_total"].fillna(df["first_order_total"].median())
df["first_order_date"]  = pd.to_datetime(df["first_order_date"]).dt.tz_localize(None)
df["days_since_order"]  = (pd.Timestamp("today").normalize() - df["first_order_date"]).dt.days

#HEURISTIC RISK SCORE (0-8) 
df["score"] = (
      2*(df["has_cancel_tag"]==1)
    + 1*(df["product_issue"] ==1)
    + 1*(df["has_urgent_tag"]==1)
    + 1*(df["num_tickets"]   >=2)
    + 1*(df["refunded"]      ==1)
    + 1*(df["utm_risk_flag"] ==1)
    + 1*(df["first_order_total"]<40)
    + 1*(df["days_since_order"]>120)
)

# VISUALISE & EXPORT 
plt.figure(figsize=(8,5))
sns.histplot(df["score"], bins=range(df["score"].max()+2), color="teal")
plt.title("Risk-Score Distribution (One-Time Customers)")
plt.xlabel("Total Risk Score"); plt.ylabel("Customers"); plt.grid(axis="y")
plt.tight_layout(); plt.show()

top10_cut = df["score"].quantile(0.9)
high_risk = df[df["score"] >= top10_cut]
high_risk.to_csv("high_risk_one_time_customers.csv", index=False)
print(f"✔  Exported {len(high_risk)} high-risk one-time customers ➜ high_risk_one_time_customers.csv")




# CREATE TARGET  (1 = score ≥ 2  → "High-Risk" one-timer)

df["high_risk_flag"] = (df["score"] >= 2).astype(int)

print("Label check (0 vs 1):")
print(df["high_risk_flag"].value_counts())  


# BUILD FEATURE MATRIX  

y = df["high_risk_flag"]

X = df.drop(columns=[
    "high_risk_flag",      
    "score",               
    "email_hash",          
    "first_order_date",    
    "customer_id"          
])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer          #  ← new
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# X / y already built 
cat_cols = ['product_form']                       
num_cols = X.select_dtypes(include=['number','bool']).columns.tolist()

# numeric pipeline: median-impute → scale 
num_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

# categorical pipeline: fill 'missing' → one-hot
cat_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ]
)

logit_clf = Pipeline(steps=[
    ('prep', preprocess),
    ('clf',  LogisticRegression(max_iter=400,
                                class_weight='balanced',
                                solver='lbfgs'))
])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

logit_clf.fit(X_train, y_train)
val_probs = logit_clf.predict_proba(X_val)[:, 1]

print(f"\nAUC (validation): {roc_auc_score(y_val, val_probs):.3f}")
print("\nClassification report (threshold 0.5):")
print(classification_report(y_val, val_probs > 0.5))



###################

#score everyone
df['subscribe_prob'] = logit_clf.predict_proba(X)[:,1]

# 2) export top 10 %
cut = df['subscribe_prob'].quantile(0.90)
df[df['subscribe_prob']>=cut] \
   .sort_values('subscribe_prob', ascending=False) \
   .to_csv('top10pct_risk_list.csv', index=False)




#  XGBOOST + SHAP  (probability model for high-risk flag)

#pip install xgboost shap

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import shap, matplotlib.pyplot as plt
import joblib                                    

# reuse the same num_cols / cat_cols lists
# (they were defined just above the logistic model)
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocess = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ]
)

xgb = XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    scale_pos_weight= y_train.value_counts()[0] / y_train.value_counts()[1],
    random_state=42,
    n_jobs=-1
)

xgb_clf = Pipeline(steps=[("prep", preprocess), ("clf", xgb)])

# train / evaluate 
xgb_clf.fit(X_train, y_train)
xgb_probs = xgb_clf.predict_proba(X_val)[:, 1]

print(f"\nXGBoost  AUC (val):  {roc_auc_score(y_val, xgb_probs):.3f}")
print(classification_report(y_val, xgb_probs > 0.5))

# SHAP  summary plot
print("Computing SHAP values… this may take ~30 s")
explainer = shap.TreeExplainer(xgb_clf.named_steps["clf"])
# transform validation set once
X_val_enc = xgb_clf.named_steps["prep"].transform(X_val)
shap_vals = explainer.shap_values(X_val_enc)


#############feature justification#############

# =============================================================
#  FEATURE-JUSTIFICATION  — covers ALL engineered features
# =============================================================
import pandas as pd, numpy as np
from collections import defaultdict

#helper: high-risk lift 
baseline_risk = df['high_risk_flag'].mean()           
def col_lift(series):
    """Return risk-rate ÷ baseline using median split for numerics."""
    if series.nunique() <= 1:             # constant col
        return np.nan
    flag = (series > series.median()) if series.dtype != 'uint8' else (series == 1)
    return round(df.loc[flag, 'high_risk_flag'].mean() / baseline_risk, 2)

#extract SHAP importance names 
enc_num_names = preprocess.named_transformers_['num'] \
                           .get_feature_names_out(num_cols)
enc_cat_names = preprocess.named_transformers_['cat'] \
                           .named_steps['onehot'] \
                           .get_feature_names_out(cat_cols)
encoded_names = list(enc_num_names) + list(enc_cat_names)
abs_shap = np.abs(shap_vals).mean(axis=0)
shap_imp = dict(zip(encoded_names, abs_shap))

# build justification rows
rows = []

# —— numeric (raw) columns
for raw in num_cols:
    rows.append({
        "Feature": raw,
        "Source": "orders" if raw.startswith("first_") else "derived",
        "Lift vs baseline": col_lift(df[raw]),
        "Mean |SHAP|": round(shap_imp.get(raw, 0), 4)
    })

#  categorical one-hot columns 
for encoded in enc_cat_names:
    # works for 'product_form=Chew'  OR  'product_form_Chew'
    if "=" in encoded:
        raw_col, cat_val = encoded.split("=", 1)
    elif "_" in encoded:
        raw_col, cat_val = encoded.split("_", 1)
    else:
        # fallback: treat the whole string as raw_col; skip lift
        raw_col, cat_val = encoded, None

    # lift: only if we could identify raw column & category
    if cat_val is not None and raw_col in df.columns:
        flag_series = (df[raw_col] == cat_val).astype(int)
        lift_val = col_lift(flag_series)
    else:
        lift_val = np.nan

    shap_score = round(shap_imp.get(encoded, 0), 4)

    rows.append({
        "Feature": encoded,
        "Source": "orders",
        "Lift vs baseline": lift_val,
        "Mean |SHAP|": shap_score
    })

#assemble & save 
justif_df = (
    pd.DataFrame(rows)
      .sort_values("Mean |SHAP|", ascending=False)
)

justif_df.to_csv("feature_justification_full.csv", index=False)
display(justif_df.head(25))                # top 25 for quick view
print("\nSaved ➜ feature_justification_full.csv")









shap.summary_plot(
    shap_vals, X_val_enc,
    feature_names = (
        list(preprocess.named_transformers_["num"].get_feature_names_out(num_cols))
        + list(preprocess.named_transformers_["cat"]
               .named_steps["onehot"].get_feature_names_out(cat_cols))
    ),
    max_display = 15,
    show=False      
)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
plt.show()
print("SHAP summary saved ➜ shap_summary.png")

# score full cohort & export 
df["subscribe_prob_xgb"] = xgb_clf.predict_proba(X)[:, 1]
df.sort_values("subscribe_prob_xgb", ascending=False) \
  .head(1000) \
  .to_csv("top1000_risk_xgb.csv", index=False)
print("Top-1 000 scored customers exported ➜ top1000_risk_xgb.csv")

# save the fitted pipeline 
joblib.dump(xgb_clf, "xgb_highrisk_pipe.pkl")
print("Saved model ➜ xgb_highrisk_pipe.pkl")
