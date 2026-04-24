import pandas as pd
from xgboost import XGBClassifier

from features import load_and_combine_symbols
from targets import build_target_pooled


# columns that are NOT model features (OHLCV, identifiers, labels)
NON_FEATURE_COLS = {
    'open', 'high', 'low', 'close', 'volume',
    'ticker', 'target', 'target_direction',
}


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def prepare_data(data_dir, timeframe, horizon, fee_threshold):
    data = load_and_combine_symbols(data_dir=data_dir, timeframe=timeframe)
    data['target'] = build_target_pooled(df=data, horizon=horizon, fee_threshold=fee_threshold)
    data = data.dropna(subset=feature_columns(data) + ['target'])
    return data


def generate_walkforward_folds(df, initial_train_years, test_months):
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


def walkforward_evaluate(df, initial_train_years, test_months):
    feature_cols = feature_columns(df)
    all_predictions = []

    folds = list(generate_walkforward_folds(df, initial_train_years, test_months))

    for fold_idx, (train_end, test_start, test_end) in enumerate(folds, 1):
        train = df[df.index < train_end]
        test  = df[(df.index >= test_start) & (df.index < test_end)]

        if len(test) == 0:
            continue

        X_train = train[feature_cols]
        y_train_mag = train['target'].astype(int)           # magnitude
        y_train_dir = train['target_direction'].astype(int) # direction

        X_test  = test[feature_cols]
        y_test_mag = test['target'].astype(int)
        y_test_dir = test['target_direction'].astype(int)

        scale_pos_weight_mag = (y_train_mag == 0).sum() / (y_train_mag == 1).sum()
        scale_pos_weight_dir = (y_train_dir == 0).sum() / (y_train_dir == 1).sum()

        def make_model(spw):
            return XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                scale_pos_weight=spw, random_state=42,
                eval_metric='logloss', n_jobs=-1,
            )

        mag_model = make_model(scale_pos_weight_mag).fit(X_train, y_train_mag)
        dir_model = make_model(scale_pos_weight_dir).fit(X_train, y_train_dir)

        p_mag = mag_model.predict_proba(X_test)[:, 1]
        p_dir = dir_model.predict_proba(X_test)[:, 1]

        fold_preds = pd.DataFrame({
            'y_true_magnitude': y_test_mag.values,
            'y_true_direction': y_test_dir.values,
            'p_magnitude': p_mag,
            'p_direction': p_dir,
            'ticker': test['ticker'].values,
            'close': test['close'].values,
            'fold': fold_idx,
        }, index=X_test.index)

        all_predictions.append(fold_preds)

    return pd.concat(all_predictions)
