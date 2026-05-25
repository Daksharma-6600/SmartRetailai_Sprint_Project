import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("dataset/train.csv")
print("Dataset Loaded Successfully")


# =========================
# DATE FEATURES
# =========================
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

df["Order Month"] = df["Order Date"].dt.month
df["Order Year"] = df["Order Date"].dt.year
df["Order Day"] = df["Order Date"].dt.day

df["Ship Delay Days"] = (df["Ship Date"] - df["Order Date"]).dt.days


# =========================
# DROP UNNECESSARY COLUMNS
# =========================
df = df.drop(columns=[
    "Row ID",
    "Order ID",
    "Customer ID",
    "Customer Name",
    "Product ID",
    "Product Name",
    "Order Date",
    "Ship Date",
    "Postal Code"
])


# =========================
# HANDLE MISSING VALUES
# =========================
df = df.dropna()


# =========================
# ONE HOT ENCODING
# =========================
df = pd.get_dummies(df, drop_first=True)


# =========================
# FEATURES & TARGET
# =========================
X = df.drop("Sales", axis=1)
y = np.log1p(df["Sales"])   # log transform improves accuracy


# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Train Test Split Done")


# =========================
# MODEL TRAINING (GOOD BASE MODEL)
# =========================
model = RandomForestRegressor(
    n_estimators=400,
    max_depth=25,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model Training Completed")


# =========================
# PREDICTION
# =========================
predictions = model.predict(X_test)

# reverse log transform
predictions = np.expm1(predictions)
y_test_actual = np.expm1(y_test)


# =========================
# EVALUATION
# =========================
mae = mean_absolute_error(y_test_actual, predictions)
r2 = r2_score(y_test_actual, predictions)

print("\nModel Evaluation")
print("----------------------")
print("MAE :", mae)
print("R2 Score :", r2)


# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "models/sales_forecast_model.pkl")

# SAVE TRAINING COLUMNS
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

print("\nModel Saved Successfully")