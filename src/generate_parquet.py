import pandas as pd
from datetime import timezone

df = pd.read_csv("data/PEDE2024-silver.csv")

df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)

print(df.columns)
print(df.head())

df.to_parquet('data/parquet/PEDE2024-silver.parquet')