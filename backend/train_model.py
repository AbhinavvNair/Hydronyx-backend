import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# --- Load data ---
rainfall = pd.read_csv("data/rainfall.csv")
groundwater = pd.read_csv("data/groundwater.csv")

# --- Merge rainfall + groundwater on state and year_month ---
df = pd.merge(
    groundwater,
    rainfall,
    on=["state_name", "year_month"],
    how="inner"
)

# --- Create lag feature (previous month's groundwater level) ---
df["lag_gw"] = df.groupby("state_name")["gw_level_m_bgl"].shift(1)
df = df.dropna()

# --- Features (X) and target (y) ---
X = df[["rainfall_actual_mm", "lag_gw"]]
y = df["gw_level_m_bgl"]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train linear regression model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Save model ---
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/groundwater_predictor.pkl")

print("✅ Model training complete. Model saved at: models/groundwater_predictor.pkl")