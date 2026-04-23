import numpy as np
import pandas as pd
import ta

def add_price_features(df):
    df = df.copy()

    df['return_1h'] = df['close'].pct_change(1)
    df['return_6h'] = df['close'].pct_change(6)
    df['return_24h'] = df['close'].pct_change(24)
    df['return_72h'] = df['close'].pct_change(72)

    df['log_return_1h'] = np.log(df['close'] / df['close'].shift(1))

    df['price_to_ma_24h'] = df['close'] / df['close'].rolling(24).mean()
    df['price_to_ma_168h'] = df['close'] / df['close'].rolling(168).mean()

    df['range_position_24h'] = ((df['close'] - df['low'].rolling(24).min()) 
                                / (df['high'].rolling(24).max() - df['low'].rolling(24).min()))

    return df


def add_volume_features(df):
    df = df.copy()

    df['volume_ratio_24h'] = df['volume'] / df['volume'].rolling(24).mean()
    df['volume_ratio_168h'] = df['volume'] / df['volume'].rolling(168).mean()

    df['signed_volume_ratio_24h'] = (
    df['volume_ratio_24h'] * np.sign(df['return_1h'])).rolling(24).mean()

    df['volume_change_1h'] = np.log(df['volume'] / df['volume'].shift(1))

    return df


def add_volatility_features(df):
    df = df.copy()

    df['volatility_24h']  = df['return_1h'].rolling(24).std()
    df['volatility_72h']  = df['return_1h'].rolling(72).std()
    df['volatility_168h'] = df['return_1h'].rolling(168).std()

    df['vol_regime_24h'] = df['volatility_24h'] / df['volatility_168h']

    df['abs_return_1h'] = np.abs(df['return_1h'])
    
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
