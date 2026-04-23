import pandas as pd
from pathlib import Path

from features import load_and_combine_symbols
from targets import build_target_pooled

combined = load_and_combine_symbols('data/raw', timeframe='1h')
combined['target'] = build_target_pooled(combined, horizon=1, fee_threshold=0.002)

print("Target stats per ticker:")
print(combined.groupby('ticker')['target'].agg(['count', 'sum', 'mean']))
print()
print("Checking last row per ticker (should be <NA>):")
for ticker, group in combined.groupby('ticker'):
    last = group.iloc[-1]
    print(f"  {ticker}: close={last['close']:.2f}, target={last['target']}")