from data_collection import load_ohlcv
from features import add_price_features, add_volume_features, add_volatility_features, add_indicator_features
from pathlib import Path

import pandas as pd

btc = load_ohlcv(Path('data/raw/BTC_USD_1h.parquet'))
btc = add_price_features(btc)
btc = add_volume_features(btc)
btc = add_volatility_features(btc)
btc = add_indicator_features(btc)

print(btc.columns.tolist())
print()
print(btc[['rsi_14', 'macd_hist', 'bb_percent', 'bb_width']].describe())