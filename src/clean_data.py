
from pathlib import Path
import pandas as pd

# ------------------------------------------------
# 1. Define paths
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "restaurant_sales_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "restaurant_sales_cleaned.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# 2. Load original dataset
# ------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)

# Keep the original CSV unchanged.


# ------------------------------------------------
# 3. Standardize column names
# ------------------------------------------------

df.columns = df.columns.str.strip().str.lower().str.replace(
    " ", "_", regex=False
)

print("\nStandardized columns:")
print(df.columns.tolist())


# ------------------------------------------------
# 4. Convert data types
# ------------------------------------------------

df["order_date"] = pd.to_datetime(
    df["order_date"], errors="coerce"
)

for col in ["price", "quantity", "order_total"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ------------------------------------------------
# 5. Check duplicate order IDs
# ------------------------------------------------

duplicate_orders = df["order_id"].duplicated().sum()

print("\nDuplicate order IDs:", duplicate_orders)

# Do not remove records silently.
# We already confirmed that the original IDs are unique.


# ------------------------------------------------
# 6. Handle missing text values
# ------------------------------------------------

text_columns = [
    "item",
    "payment_method",
]

for col in text_columns:
    df[col] = df[col].fillna("Unknown")

# We do not invent missing item names or payment methods.


# ------------------------------------------------
# 7. Validate numerical fields
# ------------------------------------------------

invalid_price = df["price"].notna() & (df["price"] < 0)
invalid_quantity = df["quantity"].notna() & (df["quantity"] <= 0)
invalid_total = df["order_total"].notna() & (df["order_total"] < 0)

print("\nInvalid prices:", invalid_price.sum())
print("Invalid quantities:", invalid_quantity.sum())
print("Invalid totals:", invalid_total.sum())

# Flag invalid values as missing, preserving the records.
df.loc[invalid_price, "price"] = pd.NA
df.loc[invalid_quantity, "quantity"] = pd.NA
df.loc[invalid_total, "order_total"] = pd.NA


# ------------------------------------------------
# 8. Recover missing totals where possible
# ------------------------------------------------

can_calculate_total = (
    df["order_total"].isna()
    & df["price"].notna()
    & df["quantity"].notna()
)

recovered_totals = can_calculate_total.sum()

df.loc[can_calculate_total, "order_total"] = (
    df.loc[can_calculate_total, "price"]
    * df.loc[can_calculate_total, "quantity"]
)

print("\nTotals recovered:", recovered_totals)


# ------------------------------------------------
# 9. Create date-based features
# ------------------------------------------------

df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.month
df["day_of_week"] = df["order_date"].dt.day_name()
df["is_weekend"] = (
    df["order_date"].dt.dayofweek >= 5
).astype("int8")


# ------------------------------------------------
# 10. Add a data completeness indicator
# ------------------------------------------------

df["purchase_details_complete"] = (
    df["item"].ne("Unknown")
    & df["price"].notna()
    & df["quantity"].notna()
    & df["order_total"].notna()
)

print(
    "\nComplete purchase-detail records:",
    df["purchase_details_complete"].sum()
)


# ------------------------------------------------
# 11. Save cleaned dataset
# ------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\n========== CLEANING SUMMARY ==========")
print("Original rows:", 17534)
print("Final rows:", len(df))
print("Final columns:", len(df.columns))

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nSaved cleaned dataset to:")
print(OUTPUT_FILE)
