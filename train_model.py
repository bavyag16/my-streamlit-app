import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib
import os

# ---------------- 1. LOAD DATASET ---------------- #
df = pd.read_csv("data/telco_churn.csv")

# ---------------- 2. CLEAN COLUMN NAMES ---------------- #
df.columns = df.columns.str.strip().str.lower()

print("Columns:", df.columns)

# ---------------- 3. DROP UNNECESSARY COLUMNS ---------------- #
if "customerid" in df.columns:
    df.drop("customerid", axis=1, inplace=True)

if "unnamed: 0" in df.columns:
    df.drop("unnamed: 0", axis=1, inplace=True)

# ---------------- 4. CLEAN TOTALCHARGES ---------------- #
if "totalcharges" in df.columns:
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df["totalcharges"] = df["totalcharges"].fillna(df["totalcharges"].median())

# ---------------- 5. CLEAN CHURN ---------------- #
if "churn" not in df.columns:
    print("❌ ERROR: churn column not found!")
    exit()

df["churn"] = df["churn"].astype(str).str.strip().str.lower()

df["churn"] = df["churn"].map({
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
    "true": 1,
    "false": 0
})

if df["churn"].isnull().sum() > 0:
    print("❌ ERROR: churn still has missing values!")
    exit()

# ---------------- 6. SELECT ONLY REQUIRED FEATURES ---------------- #
df = df[[
    "tenure",
    "monthlycharges",
    "totalcharges",
    "churn"
]]

# ---------------- 7. SPLIT DATA ---------------- #
X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- 8. TRAIN MODEL ---------------- #
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ---------------- 9. PREDICTION ---------------- #
y_pred = model.predict(X_test)

# ---------------- 10. ACCURACY ---------------- #
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---------------- 11. SAVE MODEL ---------------- #
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.pkl")

print("✅ Model trained and saved successfully!")
