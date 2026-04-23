import pandas as pd

def build_target(df, horizon=1, fee_threshold=0.002):
    # future return from time t to time t+horizon
    future_return = df['close'].pct_change(horizon).shift(-horizon)
    
    # check if return is greater than that needed to be profitable after assumed fees
    target = (future_return > fee_threshold).astype('Int64')
    # prevent NA values from appearing as 0
    target[future_return.isna()] = pd.NA
    target.name = 'target'
    
    return target

def build_target_pooled(df, horizon=1, fee_threshold=0.000, ticker_col='ticker'):
    def target_for_group(close_series):
        future_return = close_series.pct_change(horizon).shift(-horizon)
        target = (future_return > fee_threshold).astype('Int64')
        target[future_return.isna()] = pd.NA
        return target
    
    return df.groupby(ticker_col)['close'].transform(target_for_group)