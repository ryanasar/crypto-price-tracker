from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from features import load_and_combine_symbols
from targets import build_target_pooled


# everything except OHLCV, ticker, and target
FEATURE_COLS = [
    'return_1h', 'return_6h', 'return_24h', 'return_72h', 'log_return_1h',
    'price_to_ma_24h', 'price_to_ma_168h', 'range_position_24h',
    'volume_ratio_24h', 'volume_ratio_168h', 'signed_volume_ratio_24h',
    'volume_change_1h',
    'volatility_24h', 'volatility_72h', 'volatility_168h',
    'vol_regime_24h', 'abs_return_1h',
    'rsi_14', 'macd_hist', 'bb_percent', 'bb_width',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
]

def prepare_data(data_dir='data/raw', horizon=1, fee_threshold=0.002):
    data = load_and_combine_symbols(data_dir=data_dir)

    data['target'] = build_target_pooled(df=data, horizon=horizon, fee_threshold=fee_threshold)

    # remove NaN rows from feature and target data
    data = data.dropna(subset=FEATURE_COLS + ['target'])

    return data


def split_train_test(df, test_fraction=0.2):
    unique_ts = df.index.unique().sort_values()
    cutoff = unique_ts[int(len(unique_ts) * (1 - test_fraction))]

    train = df.loc[df.index < cutoff]
    test = df.loc[df.index >= cutoff]

    return train, test

if __name__ == '__main__':
    df = prepare_data()
    train, test = split_train_test(df, test_fraction=0.2)
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print(f"\nTrain range: {train.index.min()} to {train.index.max()}")
    print(f"Test range:  {test.index.min()} to {test.index.max()}")
    print(f"\nTrain positive fraction: {train['target'].mean():.3f}")
    print(f"Test  positive fraction: {test['target'].mean():.3f}")
    print(f"\nTrain rows per ticker:")
    print(train.groupby('ticker').size())
    print(f"\nTest rows per ticker:")
    print(test.groupby('ticker').size())