import pandas as pd


def build_target_pooled(df, horizon=1, fee_threshold=0.002, ticker_col='ticker'):
    def target_for_group(close_series):
        future_return = close_series.pct_change(horizon).shift(-horizon)
        target = (future_return > fee_threshold).astype('Int64')
        target[future_return.isna()] = pd.NA
        return target

    return df.groupby(ticker_col)['close'].transform(target_for_group)
