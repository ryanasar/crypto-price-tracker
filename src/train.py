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


def train_model(train, test, feature_cols=FEATURE_COLS):
    x_train = train[feature_cols]
    y_train = train['target'].astype(int)

    x_test = test[feature_cols]
    y_test = test['target'].astype(int)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    y_prob = model.predict_proba(x_test)[:, 1]

    predictions = pd.DataFrame({
        'y_true': y_test.values,
        'y_prob': y_prob,
    }, index=x_test.index)

    return model, predictions


if __name__ == '__main__':
    df = prepare_data()
    train, test = split_train_test(df, test_fraction=0.2)
    model, predictions = train_model(train, test)

    test_with_pred = test.copy()
    test_with_pred['y_prob'] = predictions['y_prob'].values

    from sklearn.metrics import roc_auc_score

    print("\nAUC per ticker (test set):")
    for ticker in test_with_pred['ticker'].unique():
        subset = test_with_pred[test_with_pred['ticker'] == ticker]
        auc = roc_auc_score(subset['target'], subset['y_prob'])
        pos_rate = subset['target'].mean()
        print(f"  {ticker}: AUC = {auc:.4f}, positive rate = {pos_rate:.4f}, n = {len(subset)}")