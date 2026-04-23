import numpy as np
import pandas as pd

def add_price_features(df):
    df = df.copy()
    """
    Add price-derived features to a single-symbol OHLCV DataFrame.
    Returns a new DataFrame with the original columns plus:
        return_1h, return_6h, return_24h, return_72h
        log_return_1h
        price_to_ma_24, price_to_ma_168
        range_position_24
    """

    df['return_1h'] = df['close'].pct_change(1)
    df['return_6h'] = df['close'].pct_change(6)
    df['return_24h'] = df['close'].pct_change(24)
    df['return_72h'] = df['close'].pct_change(72)

    df['log_return_1h'] = np.log(df['close'] / df['close'].shift(1))

    df['price_to_ma_24h'] = df['close'] / df['close'].rolling(24).mean()
    df['price_to_ma_168h'] = df['close'] / df['close'].rolling(168).mean()

    df['range_position_24h'] = (df['close'] - df['low'].rolling(24).min()) / (df['high'].rolling(24).max() - df['low'].rolling(24).min())

    return df

    




