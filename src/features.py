from pathlib import Path
import numpy as np
import pandas as pd
from data_collection import load_ohlcv
import ta

def add_price_features(df):
    df = df.copy()

    df['return_1'] = df['close'].pct_change(1)
    df['return_6'] = df['close'].pct_change(6)
    df['return_24'] = df['close'].pct_change(24)
    df['return_72'] = df['close'].pct_change(72)

    df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))

    df['price_to_ma_24'] = df['close'] / df['close'].rolling(24).mean()
    df['price_to_ma_168'] = df['close'] / df['close'].rolling(168).mean()

    df['range_position_24'] = ((df['close'] - df['low'].rolling(24).min()) 
                                / (df['high'].rolling(24).max() - df['low'].rolling(24).min()))

    return df


def add_volume_features(df):
    df = df.copy()

    df['volume_ratio_24'] = df['volume'] / df['volume'].rolling(24).mean()
    df['volume_ratio_168'] = df['volume'] / df['volume'].rolling(168).mean()

    df['signed_volume_ratio_24'] = (
    df['volume_ratio_24'] * np.sign(df['return_1'])).rolling(24).mean()

    df['volume_change_1'] = np.log(df['volume'] / df['volume'].shift(1))

    return df


def add_volatility_features(df):
    df = df.copy()

    df['volatility_24']  = df['return_1'].rolling(24).std()
    df['volatility_72']  = df['return_1'].rolling(72).std()
    df['volatility_168'] = df['return_1'].rolling(168).std()

    df['vol_regime_24'] = df['volatility_24'] / df['volatility_168']

    df['abs_return_1'] = np.abs(df['return_1'])
    
    return df


def add_indicator_features(df):
    df = df.copy()

    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)

    df['macd_hist'] = ta.trend.macd_diff(df['close']) / df['close']

    df['bb_percent'] = ta.volatility.bollinger_pband(df['close'])
    df['bb_width'] = ta.volatility.bollinger_wband(df['close'])
    
    return df


def add_time_features(df):
    df = df.copy()

    # encode hour and day to account for cycle wrapping
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)  

    return df


def build_features(df):
    df = add_price_features(df)
    df = add_volume_features(df)
    df = add_volatility_features(df)
    df = add_indicator_features(df)
    df = add_time_features(df)

    return df

def load_and_combine_symbols(data_dir, timeframe='1h'):
    data_dir = Path(data_dir)
    all_dfs = []
    
    for path in sorted(data_dir.glob(f'*_{timeframe}.parquet')):
        # extract ticker from filename
        ticker = path.stem.rsplit('_', 1)[0]
        
        df = load_ohlcv(path)
        df = build_features(df)
        df['ticker'] = ticker
        
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs)
    combined = combined.sort_index()
    return combined
