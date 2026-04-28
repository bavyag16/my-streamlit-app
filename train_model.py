import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# 1. Load dataset
df = pd.read_csv("data/telco_churn.csv")

# 2. Standardize column names
df.columns = df.columns.str.strip().str.lower()

print("Columns:", df.columns)

# 3. Drop unnecessary columns
if "customerid" in df.columns:
    df.drop("customerid", axis=1, inplace=True)

if "unnamed: 0" in df.columns:
    df.drop("unnamed: 0", axis=1, inplace=True)

# 4. Fix TotalCharges
if "totalcharges" in df.columns:
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df["totalcharges"] = df["totalcharges"].fillna(df["totalcharges"].median())

# 5. Fix Churn column (handle all formats)
if "churn" not in df.columns:
    print("❌ ERROR: 'churn' column not found!")
    exit()

print("Original Churn values:", df["churn"].unique())

df["churn"] = df["churn"].astype(str).str.strip().str.lower()

df["churn"] = df["churn"].map({
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
    "true": 1,
    "false": 0
})

print("Mapped Churn values:", df["churn"].unique())

if df["churn"].isnull().sum() > 0:
    print("❌ Error: Churn still has NaN values!")
    exit()

# 6. Convert categorical columns (IMPORTANT FIX)
df = pd.get_dummies(df, drop_first=True)

# 7. Split data
X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train model
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# 9. Prediction
y_pred = model.predict(X_test)

# 10. Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 11. Save model
joblib.dump(model, "models/churn_model.pkl")

print("✅ Model saved successfully!")
