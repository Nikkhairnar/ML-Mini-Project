from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR / "data" / "processed"
    / "restaurant_sales_cleaned.csv"
)

df = pd.read_csv(DATA_FILE, parse_dates=["order_date"])

sales = df[
    df["purchase_details_complete"].astype(str).str.lower()
    .isin(["true", "1"])
].copy()

daily = sales.groupby("order_date").agg(
    daily_sales=("order_total", "sum"),
    daily_units=("quantity", "sum"),
).sort_index()

calendar = pd.date_range(
    daily.index.min(), daily.index.max(), freq="D"
)

daily = daily.reindex(calendar).fillna(0)
daily.index.name = "order_date"

# Match the feature preparation in demand_prediction.py.
for target in ["daily_sales", "daily_units"]:
    daily[f"{target}_lag_1"] = daily[target].shift(1)
    daily[f"{target}_lag_7"] = daily[target].shift(7)
    daily[f"{target}_lag_14"] = daily[target].shift(14)
    daily[f"{target}_rolling_7"] = (
        daily[target].shift(1).rolling(7).mean()
    )

daily = daily.dropna()

# Match the exact chronological 80/20 split.
split_index = int(len(daily) * 0.8)
train = daily.iloc[:split_index].copy()
test = daily.iloc[split_index:].copy()

results = []

for target in ["daily_sales", "daily_units"]:
    actual = test[target]
    valid_idx = actual > 0

    # 1. Seasonal-naive baseline (same day last week)
    pred_seasonal = test[f"{target}_lag_7"]
    mae_s = mean_absolute_error(actual, pred_seasonal)
    rmse_s = np.sqrt(mean_squared_error(actual, pred_seasonal))
    r2_s = r2_score(actual, pred_seasonal)
    mape_s = np.mean(np.abs((actual[valid_idx] - pred_seasonal[valid_idx]) / actual[valid_idx])) * 100
    acc_s = max(0.0, 100.0 - mape_s)

    results.append({
        "target": target,
        "baseline_model": "Weekly Seasonal-Naive (Lag-7)",
        "MAE": round(mae_s, 2),
        "RMSE": round(rmse_s, 2),
        "MAPE (%)": round(mape_s, 2),
        "Accuracy (%)": round(acc_s, 2),
        "Test R2": round(r2_s, 4),
    })

    # 2. Historical Mean baseline
    mean_val = train[target].mean()
    pred_mean = np.full(len(test), mean_val)
    mae_m = mean_absolute_error(actual, pred_mean)
    rmse_m = np.sqrt(mean_squared_error(actual, pred_mean))
    r2_m = r2_score(actual, pred_mean)
    mape_m = np.mean(np.abs((actual[valid_idx] - pred_mean[valid_idx]) / actual[valid_idx])) * 100
    acc_m = max(0.0, 100.0 - mape_m)

    results.append({
        "target": target,
        "baseline_model": "Historical Train Mean",
        "MAE": round(mae_m, 2),
        "RMSE": round(rmse_m, 2),
        "MAPE (%)": round(mape_m, 2),
        "Accuracy (%)": round(acc_m, 2),
        "Test R2": round(r2_m, 4),
    })

print("Test period:", test.index.min().date(), "to", test.index.max().date())
print("\n" + "=" * 55)
print("BENCHMARK BASELINE EVALUATION")
print("=" * 55)
print(pd.DataFrame(results).to_string(index=False))
