from data_collection import load_ohlcv
from features import *
from pathlib import Path

import pandas as pd

btc = load_ohlcv(Path('data/raw/BTC_USD_1h.parquet'))
btc = add_time_features(btc)

print(btc[['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']].describe())
print()
print(btc[['hour_sin', 'hour_cos']].head(26))