
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# 1. Load cleaned data
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "restaurant_sales_cleaned.csv"
)

OUTPUT_DIR = BASE_DIR / "reports" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FILE, parse_dates=["order_date"])

# Use only records with complete purchase details
sales = df[df["purchase_details_complete"]].copy()

sns.set_theme(style="whitegrid")

print("Total records:", len(df))
print("Complete purchase records:", len(sales))
print("Unique customers:", df["customer_id"].nunique())

# ---------------------------------------
# 2. Overall KPIs
# ---------------------------------------

print("\n========== OVERALL SALES ==========")

print("Total sales:", round(sales["order_total"].sum(), 2))
print("Total units sold:", int(sales["quantity"].sum()))
print("Average order value:", round(sales["order_total"].mean(), 2))
print("Unique customers:", sales["customer_id"].nunique())
print("Average quantity per record:", round(sales["quantity"].mean(), 2))

# ---------------------------------------
# 3. Monthly sales
# ---------------------------------------

sales["month_period"] = (
    sales["order_date"].dt.to_period("M")
)

monthly_sales = (
    sales.groupby("month_period")["order_total"]
    .sum()
)

print("\n========== MONTHLY SALES ==========")
print(monthly_sales)

plt.figure(figsize=(12, 5))
monthly_sales.index = monthly_sales.index.astype(str)
monthly_sales.plot(marker="o")
plt.title("Monthly Restaurant Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_sales.png", dpi=150)
plt.close()

# ---------------------------------------
# 4. Sales by category
# ---------------------------------------

category_sales = (
    sales.groupby("category")["order_total"]
    .sum()
    .sort_values(ascending=False)
)

print("\n========== SALES BY CATEGORY ==========")
print(category_sales)

plt.figure(figsize=(9, 5))
sns.barplot(
    x=category_sales.values,
    y=category_sales.index
)
plt.title("Sales by Category")
plt.xlabel("Sales")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "category_sales.png", dpi=150)
plt.close()

# ---------------------------------------
# 5. Most popular items by quantity
# ---------------------------------------

popular_items = (
    sales.groupby("item")["quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n========== TOP 10 ITEMS BY QUANTITY ==========")
print(popular_items)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=popular_items.values,
    y=popular_items.index
)
plt.title("Top 10 Items by Quantity Sold")
plt.xlabel("Units Sold")
plt.ylabel("Item")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "popular_items.png", dpi=150)
plt.close()

# ---------------------------------------
# 6. Customer behaviour
# ---------------------------------------

customer_stats = (
    sales.groupby("customer_id")
    .agg(
        purchase_records=("order_id", "count"),
        total_spending=("order_total", "sum"),
        total_units=("quantity", "sum"),
        average_order_value=("order_total", "mean"),
    )
    .sort_values("total_spending", ascending=False)
)

print("\n========== TOP 10 CUSTOMERS BY SPENDING ==========")
print(customer_stats.head(10))

customer_stats.to_csv(
    OUTPUT_DIR / "customer_summary.csv"
)

# ---------------------------------------
# 7. Payment methods
# ---------------------------------------

payment_counts = (
    df["payment_method"]
    .value_counts()
)

print("\n========== PAYMENT METHODS ==========")
print(payment_counts)

# ---------------------------------------
# 8. Sales by day of week
# ---------------------------------------

day_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

sales["day_of_week"] = sales["order_date"].dt.day_name()

weekday_sales = (
    sales.groupby("day_of_week")["order_total"]
    .sum()
    .reindex(day_order)
)

print("\n========== SALES BY DAY ==========")
print(weekday_sales)

plt.figure(figsize=(10, 5))
sns.barplot(
    x=weekday_sales.index,
    y=weekday_sales.values
)
plt.title("Sales by Day of Week")
plt.xlabel("Day")
plt.ylabel("Sales")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "weekday_sales.png", dpi=150)
plt.close()

print("\nEDA completed.")
print("Charts saved in:", OUTPUT_DIR)
