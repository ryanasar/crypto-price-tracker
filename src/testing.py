import pandas as pd
from pathlib import Path

for symbol_file in ['BTC_USD_1h.parquet', 'ETH_USD_1h.parquet']:
    df = pd.read_parquet(Path('data/raw') / symbol_file)
    print(f"\n=== {symbol_file} ===")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Index type: {type(df.index).__name__}")
    print(f"Timezone: {df.index.tz}")
    print(f"Duplicates: {df.index.duplicated().sum()}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head(3))
    print(f"\nLast few rows:")
    print(df.tail(3))