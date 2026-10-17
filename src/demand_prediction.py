from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# -------------------------------------------
# 1. Paths
# -------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "restaurant_sales_cleaned.csv"
)

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "eda"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------
# 2. Load data
# -------------------------------------------

df = pd.read_csv(DATA_FILE, parse_dates=["order_date"])

# Use complete purchase records only.
sales = df[
    df["purchase_details_complete"].astype(str).str.lower()
    .isin(["true", "1"])
].copy()

sales["order_total"] = pd.to_numeric(
    sales["order_total"], errors="coerce"
)
sales["quantity"] = pd.to_numeric(
    sales["quantity"], errors="coerce"
)

sales = sales.dropna(
    subset=["order_date", "order_total", "quantity"]
)

# -------------------------------------------
# 3. Aggregate by day
# -------------------------------------------

daily = (
    sales.groupby("order_date")
    .agg(
        daily_sales=("order_total", "sum"),
        daily_units=("quantity", "sum"),
        recorded_purchases=("order_id", "nunique"),
    )
    .sort_index()
)

# Continuous daily calendar
calendar = pd.date_range(
    daily.index.min(),
    daily.index.max(),
    freq="D",
)

daily = daily.reindex(calendar)
daily.index.name = "order_date"

daily[[
    "daily_sales",
    "daily_units",
    "recorded_purchases",
]] = daily[[
    "daily_sales",
    "daily_units",
    "recorded_purchases",
]].fillna(0)

# -------------------------------------------
# 4. Create time-based and lag features
# -------------------------------------------

daily["day_of_week"] = daily.index.dayofweek
daily["month"] = daily.index.month
daily["day_of_month"] = daily.index.day
daily["is_weekend"] = (
    daily["day_of_week"] >= 5
).astype(int)

# Lag and rolling features (shifted to strictly prevent data leakage)
for target in ["daily_sales", "daily_units"]:
    daily[f"{target}_lag_1"] = daily[target].shift(1)
    daily[f"{target}_lag_7"] = daily[target].shift(7)
    daily[f"{target}_lag_14"] = daily[target].shift(14)
    daily[f"{target}_rolling_7"] = (
        daily[target].shift(1).rolling(7).mean()
    )

daily = daily.dropna().copy()

# -------------------------------------------
# 5. Chronological train/test split (80/20)
# -------------------------------------------

feature_columns = [
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "daily_sales_lag_1",
    "daily_sales_lag_7",
    "daily_sales_lag_14",
    "daily_sales_rolling_7",
    "daily_units_lag_1",
    "daily_units_lag_7",
    "daily_units_lag_14",
    "daily_units_rolling_7",
]

split_index = int(len(daily) * 0.8)

train = daily.iloc[:split_index]
test = daily.iloc[split_index:]

X_train = train[feature_columns]
X_test = test[feature_columns]

print(f"Total modeling days: {len(daily)}")
print(f"Training days: {len(train)} ({train.index.min().date()} to {train.index.max().date()})")
print(f"Testing days: {len(test)} ({test.index.min().date()} to {test.index.max().date()})")

# -------------------------------------------
# 6. Define candidate models
# -------------------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=50.0, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    ),
}

results = []
predictions = {}

# -------------------------------------------
# 7. Train, predict and calculate metrics
# -------------------------------------------

for target in ["daily_sales", "daily_units"]:
    y_train = train[target]
    y_test = test[target]

    for model_name, model_template in models.items():
        model = model_template.__class__(
            **model_template.get_params()
        )

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        predicted = model.predict(X_test)

        mae = mean_absolute_error(y_test, predicted)
        rmse = np.sqrt(mean_squared_error(y_test, predicted))
        r2 = r2_score(y_test, predicted)
        train_r2 = r2_score(y_train, y_train_pred)

        # Business metrics: Mean Absolute Percentage Error (MAPE) & Forecast Accuracy
        # Avoid division by zero by filtering non-zero targets
        valid_idx = y_test > 0
        mape = np.mean(np.abs((y_test[valid_idx] - predicted[valid_idx]) / y_test[valid_idx])) * 100
        forecast_accuracy = max(0.0, 100.0 - mape)

        results.append({
            "target": target,
            "model": model_name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE (%)": round(mape, 2),
            "Accuracy (%)": round(forecast_accuracy, 2),
            "Test R2": round(r2, 4),
            "Train R2": round(train_r2, 4),
        })

        predictions[(target, model_name)] = predicted

        # Save model artifact
        filename = f"{target}_{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, MODEL_DIR / filename)

# -------------------------------------------
# 8. Print and save results
# -------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 55)
print("DEMAND PREDICTION MODEL EVALUATION")
print("=" * 55)
print(results_df.to_string(index=False))

results_df.to_csv(
    REPORT_DIR / "prediction_model_results.csv",
    index=False,
)

# -------------------------------------------
# 9. Save test predictions CSV
# -------------------------------------------

prediction_output = test[["daily_sales", "daily_units"]].copy()

for (target, model_name), values in predictions.items():
    column = f"{target}_{model_name.lower().replace(' ', '_')}_pred"
    prediction_output[column] = np.round(values, 2)

prediction_output.to_csv(REPORT_DIR / "daily_predictions.csv")

print(f"\nModels saved successfully in: {MODEL_DIR}")
print(f"Results and daily test predictions saved in: {REPORT_DIR}")
