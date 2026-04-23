from data_collection import load_ohlcv
from features import add_price_features, add_volume_features
from pathlib import Path

import pandas as pd

btc = load_ohlcv(Path('data/raw/BTC_USD_1h.parquet'))
btc = add_price_features(btc)
btc = add_volume_features(btc)

print(btc[['volume_ratio_24h', 'volume_ratio_168h', 'signed_volume_ratio_24h', 'volume_change_1h']].describe())