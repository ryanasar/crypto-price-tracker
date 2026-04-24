"""
Crypto price prediction pipeline.

Usage:
    python main.py                  # run with existing parquet data in DATA_DIR
    python main.py --collect-data   # fetch fresh OHLCV first, then run
"""
import argparse
from sklearn.metrics import roc_auc_score

from data_collection import collect_all_data
from features import *
from targets import *
from train import *
from backtest import *


# =============================================================================
# Configuration
# =============================================================================

# --- Data collection ---------------------------------------------------------
SYMBOLS           = ['BTC/USD', 'ETH/USD']  # exchange symbols to fetch (ccxt format)
TIMEFRAME         = '1d'                    # candle size: '1m', '5m', '1h', '1d', ...
YEARS_OF_HISTORY  = 4                       # how far back to fetch when --collect-data is passed
FETCH_BATCH_LIMIT = 300                     # rows per API call (300 is coinbase max)
DATA_DIR          = 'data/raw'              # directory where parquet files live

# --- Target labels -----------------------------------------------------------
HORIZON                  = 1                # bars ahead to predict (e.g. 24 = 24h for '1h' data)
FEE_THRESHOLD            = 0.002            # return above this = profitable trade (magnitude target)
DIRECTION_FEE_THRESHOLD  = 0.0              # return above this = "up" (direction-only target)

# --- Walk-forward evaluation -------------------------------------------------
INITIAL_TRAIN_YEARS = 2                     # size of the initial training window, in years
TEST_MONTHS         = 1                     # size of each rolling out-of-sample test slice, in months

# --- Backtest / strategy -----------------------------------------------------
UPPER_THRESHOLD  = 0.50                     # combined prob above this → go long
LOWER_THRESHOLD  = 0.30                     # combined prob below this → exit to cash
FEE_PER_SIDE     = 0.000                    # trading fee applied per side in the backtest
PERIODS_PER_YEAR = 24 * 365                 # bars per year, used for annualization (matches TIMEFRAME)

# --- Reporting ---------------------------------------------------------------
REPORT_TICKERS = [s.replace('/', '_') for s in SYMBOLS]

# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--collect-data',
        action='store_true',
        help='fetch fresh OHLCV parquet files into DATA_DIR before running the pipeline',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.collect_data:
        collect_all_data(
            symbols=SYMBOLS,
            timeframe=TIMEFRAME,
            years_back=YEARS_OF_HISTORY,
            output_dir=DATA_DIR,
            limit=FETCH_BATCH_LIMIT,
        )

    df = prepare_data(
        data_dir=DATA_DIR,
        timeframe=TIMEFRAME,
        horizon=HORIZON,
        fee_threshold=FEE_THRESHOLD,
    )
    df['target_direction'] = build_target_pooled(
        df, horizon=HORIZON, fee_threshold=DIRECTION_FEE_THRESHOLD,
    )
    df = df.dropna(subset=['target_direction'])

    predictions = walkforward_evaluate(
        df,
        initial_train_years=INITIAL_TRAIN_YEARS,
        test_months=TEST_MONTHS,
    )

    predictions['p_combined'] = predictions['p_magnitude'] * predictions['p_direction']
    predictions = predictions.dropna()

    print(f"\nMagnitude AUC: {roc_auc_score(predictions['y_true_magnitude'], predictions['p_magnitude']):.4f}")
    print(f"Direction AUC: {roc_auc_score(predictions['y_true_direction'], predictions['p_direction']):.4f}")

    for ticker in REPORT_TICKERS:
        preds = predictions[predictions['ticker'] == ticker].copy().sort_index()
        preds['y_prob'] = preds['p_combined']

        result = simulate_strategy(
            preds,
            upper_threshold=UPPER_THRESHOLD,
            lower_threshold=LOWER_THRESHOLD,
            fee_per_side=FEE_PER_SIDE,
        )

        metrics = compute_metrics(result, periods_per_year=PERIODS_PER_YEAR)

        print(f"\n=== {ticker} ===")
        print(f"Total return:      {metrics['total_return']:>8.2%}")
        print(f"Annualized ret:    {metrics['annualized_return']:>8.2%}")
        print(f"Annualized vol:    {metrics['annualized_vol']:>8.2%}")
        print(f"Sharpe:            {metrics['sharpe']:>8.3f}")
        print(f"Max drawdown:      {metrics['max_drawdown']:>8.2%}")
        print(f"Calmar:            {metrics['calmar']:>8.3f}")
        print(f"Num trades:        {metrics['num_trades']:>8d}")
        print(f"Fraction of time long: {metrics['fraction_long']:>8.2%}")
        print(f"Win rate:          {metrics['win_rate']:>8.2%}")
        print(f"Avg win:           {metrics['avg_win']:>8.4%}")
        print(f"Avg loss:          {metrics['avg_loss']:>8.4%}")
        print(f"Expectancy:        {metrics['expectancy']:>8.4%}")

        # buy and hold baseline
        bh_pnl = result['close'].pct_change()
        bh_total = result['close'].iloc[-1] / result['close'].iloc[0] - 1
        years = len(result) / PERIODS_PER_YEAR
        bh_annual = (1 + bh_total) ** (1 / years) - 1
        bh_vol = bh_pnl.std() * PERIODS_PER_YEAR ** 0.5
        bh_sharpe = bh_annual / bh_vol if bh_vol > 0 else 0
        bh_dd = (result['close'] / result['close'].expanding().max() - 1).min()

        print(f"\n  Buy-and-hold baseline:")
        print(f"  Total return:    {bh_total:>8.2%}")
        print(f"  Annualized ret:  {bh_annual:>8.2%}")
        print(f"  Annualized vol:  {bh_vol:>8.2%}")
        print(f"  Sharpe:          {bh_sharpe:>8.3f}")
        print(f"  Max drawdown:    {bh_dd:>8.2%}")
