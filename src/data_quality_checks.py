
from pathlib import Path
import pandas as pd

# Locate and load the original dataset
BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR / "data" / "restaurant_sales_data.csv"

df = pd.read_csv(FILE_PATH)

print("=" * 50)
print("1. ORDER ID CHECK")
print("=" * 50)

print("Unique order IDs:", df["Order ID"].nunique())
print("Duplicate order IDs:", df["Order ID"].duplicated().sum())

print("\n" + "=" * 50)
print("2. NUMERICAL VALUE CHECK")
print("=" * 50)

for col in ["Price", "Quantity", "Order Total"]:
    print(f"\n{col}")
    print("Minimum:", df[col].min())
    print("Maximum:", df[col].max())
    print("Missing:", df[col].isna().sum())
    print("Zero values:", (df[col] == 0).sum())
    print("Negative values:", (df[col] < 0).sum())

print("\n" + "=" * 50)
print("3. PRICE × QUANTITY VS ORDER TOTAL")
print("=" * 50)

complete = df.dropna(
    subset=["Price", "Quantity", "Order Total"]
).copy()

complete["Calculated Total"] = (
    complete["Price"] * complete["Quantity"]
)

complete["Difference"] = (
    complete["Order Total"] - complete["Calculated Total"]
).round(2)

mismatch = complete[
    complete["Difference"].abs() > 0.01
]

print("Rows with all three values:", len(complete))
print("Mismatched totals:", len(mismatch))

print("\nExamples of mismatched totals:")
print(
    mismatch[
        [
            "Order ID",
            "Price",
            "Quantity",
            "Order Total",
            "Calculated Total",
            "Difference",
        ]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 50)
print("4. MISSING VALUES TOGETHER")
print("=" * 50)

print(
    df[["Item", "Price", "Quantity", "Order Total"]]
    .isna()
    .sum(axis=1)
    .value_counts()
    .sort_index()
    .rename("Number of rows")
)

print("\n" + "=" * 50)
print("5. CATEGORY AND ITEM CHECK")
print("=" * 50)

print("\nCategory counts:")
print(df["Category"].value_counts())

print("\nMost common items:")
print(df["Item"].value_counts().head(10))

print("\nItems appearing in multiple categories:")
item_categories = (
    df.dropna(subset=["Item"])
    .groupby("Item")["Category"]
    .nunique()
)

print(item_categories[item_categories > 1])

print("\n" + "=" * 50)
print("6. DATE CHECK")
print("=" * 50)

dates = pd.to_datetime(df["Order Date"], errors="coerce")

print("Invalid dates:", dates.isna().sum())
print("Earliest date:", dates.min())
print("Latest date:", dates.max())
print("Distinct dates:", dates.nunique())

print("\nData quality checks completed.")
