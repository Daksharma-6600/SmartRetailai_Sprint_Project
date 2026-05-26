import pandas as pd
import numpy as np


# LOAD RAW DATA SAFELY

df = pd.read_csv(
    "dataset/train.csv",
    dtype=str,              # force everything as string first
    quotechar='"',
    encoding="utf-8",
    on_bad_lines="skip"
)

print("Raw data loaded:", df.shape)


# CLEAN COLUMN NAMES

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)


# CLEAN PRODUCT NAME

if "Product_Name" in df.columns:
    df["Product_Name"] = (
        df["Product_Name"]
        .astype(str)
        .str.replace('"', '', regex=False)
        .str.replace("'", "", regex=False)
        .str.strip()
    )


# CLEAN SALES COLUMN 

df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

# remove invalid sales rows
before = df.shape[0]
df = df.dropna(subset=["Sales"])
after = df.shape[0]

print(f"Removed {before - after} invalid Sales rows")


# OPTIONAL: CLEAN OTHER NUMERIC COLUMNS

numeric_cols = ["Quantity", "Discount", "Profit"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# DROP FULLY EMPTY ROWS

df = df.dropna(how="all")


# FINAL SANITY CHECK

print("\nDATA TYPES AFTER CLEANING:\n")
print(df.dtypes)

print("\nSALES SAMPLE:\n")
print(df["Sales"].head())

print("\nNULL CHECK:\n")
print(df.isnull().sum())


# SAVE CLEAN FILE 

output_path = "dataset/train_fabric_final.csv"

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8"
)

print("\n✅ Clean file saved at:", output_path)
print("Final Shape:", df.shape)