import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option("display.max_columns", None)

# Load dataset (update path as needed)
df = pd.read_csv("housing.csv")

print("\n--- HEAD ---")
print(df.head())

print("\n--- INFO ---")
print(df.info())

print("\n--- DESCRIPTION ---")
print(df.describe(include="all").T)

# Data quality checks
print("\nDuplicate rows:", df.duplicated().sum())
print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False))

# Histograms for numerical data
num = df.select_dtypes(include=np.number)
num.hist(bins=30, figsize=(12,8))
plt.tight_layout()
plt.show()