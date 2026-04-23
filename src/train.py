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


def generate_walkforward_folds(df, initial_train_years=2, test_months=1):
    start = df.index.min()
    end = df.index.max()

    train_end = start + pd.DateOffset(years=initial_train_years)
    
    while train_end < end:
        test_start = train_end
        test_end = train_end + pd.DateOffset(months=test_months)
        
        if test_end > end:
            test_end = end
        
        yield train_end, test_start, test_end
        
        train_end = test_end


def walkforward_evaluate(df, feature_cols=FEATURE_COLS, initial_train_years=2, test_months=1):
    all_predictions = []
    
    folds = list(generate_walkforward_folds(df, initial_train_years, test_months))
    
    for fold_idx, (train_end, test_start, test_end) in enumerate(folds, 1):
        train = df[df.index < train_end]
        test  = df[(df.index >= test_start) & (df.index < test_end)]
        
        if len(test) == 0:
            continue
        
        X_train = train[feature_cols]
        y_train = train['target'].astype(int)
        X_test  = test[feature_cols]
        y_test  = test['target'].astype(int)
        
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
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        
        fold_preds = pd.DataFrame({
            'y_true': y_test.values,
            'y_prob': y_prob,
            'ticker': test['ticker'].values,
            'fold': fold_idx,
        }, index=X_test.index)
        
        all_predictions.append(fold_preds)
        
        print(f"Fold {fold_idx:2d}/{len(folds)}: train={len(train):,}, test={len(test):,}, "
              f"test_range={test_start.date()} to {test_end.date()}")
    
    return pd.concat(all_predictions)


if __name__ == '__main__':
    df = prepare_data()
    
    print("Running walk-forward evaluation...")
    predictions = walkforward_evaluate(df, initial_train_years=2, test_months=1)
    
    print(f"\nTotal OOS predictions: {len(predictions):,}")
    
    # Overall metrics
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(predictions['y_true'], predictions['y_prob'] > 0.5)
    auc = roc_auc_score(predictions['y_true'], predictions['y_prob'])
    print(f"\nOverall OOS accuracy: {acc:.4f}")
    print(f"Overall OOS AUC: {auc:.4f}")
    
    # Per-ticker
    print("\nPer-ticker AUC:")
    for ticker in predictions['ticker'].unique():
        subset = predictions[predictions['ticker'] == ticker]
        auc = roc_auc_score(subset['y_true'], subset['y_prob'])
        print(f"  {ticker}: AUC = {auc:.4f}, n = {len(subset):,}")
    
    # Per-fold AUC
    print("\nPer-fold AUC:")
    for fold in sorted(predictions['fold'].unique()):
        subset = predictions[predictions['fold'] == fold]
        try:
            auc = roc_auc_score(subset['y_true'], subset['y_prob'])
            print(f"  Fold {fold:2d}: AUC = {auc:.4f}, n = {len(subset):,}")
        except ValueError:
            print(f"  Fold {fold:2d}: couldn't compute AUC (single class?)")