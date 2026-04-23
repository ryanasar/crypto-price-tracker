from data_collection import load_ohlcv
from features import add_price_features
from pathlib import Path

btc = load_ohlcv(Path('data/raw/BTC_USD_1h.parquet'))
btc = add_price_features(btc)

print(btc.columns.to_list())
print(btc.isna().sum())
print(btc.describe())