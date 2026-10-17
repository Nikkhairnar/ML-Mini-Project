
from pathlib import Path
import pandas as pd

# Locate the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the CSV
file_path = BASE_DIR / "data" / "restaurant_sales_data.csv"
df = pd.read_csv(file_path)

# Inspect the dataset
print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- BASIC STATISTICS ---")
print(df.describe(include="all").T)
