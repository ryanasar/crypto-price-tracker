import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score

from features import *
from targets import *
from train import *
from backtest import *

if __name__ == '__main__':
    df = prepare_data(fee_threshold=0.002)
    df['target_direction'] = build_target_pooled(df, horizon=1, fee_threshold=0)
    df = df.dropna(subset=['target_direction'])

    predictions = walkforward_evaluate(df, initial_train_years=2, test_months=1)

    predictions['p_combined'] = predictions['p_magnitude'] * predictions['p_direction']
    predictions = predictions.dropna()

    #print(f"\nCombined signal distribution:")
    #print(predictions['p_combined'].describe())

    print(f"\nMagnitude AUC: {roc_auc_score(predictions['y_true_magnitude'], predictions['p_magnitude']):.4f}")
    print(f"Direction AUC: {roc_auc_score(predictions['y_true_direction'], predictions['p_direction']):.4f}")

    for ticker in ['BTC_USD', 'ETH_USD']:
        preds = predictions[predictions['ticker'] == ticker].copy().sort_index()
        preds['y_prob'] = preds['p_combined']

        result = simulate_strategy(
            preds,
            upper_threshold=0.50,
            lower_threshold=0.30,
            fee_per_side=0.000,
        )

        metrics = compute_metrics(result)

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

        # for buy and hold
        bh_pnl = result['close'].pct_change()
        bh_total = result['close'].iloc[-1] / result['close'].iloc[0] - 1
        years = len(result) / (24 * 365)
        bh_annual = (1 + bh_total) ** (1 / years) - 1
        bh_vol = bh_pnl.std() * (24 * 365) ** 0.5
        bh_sharpe = bh_annual / bh_vol if bh_vol > 0 else 0
        bh_dd = (result['close'] / result['close'].expanding().max() - 1).min()

        print(f"\n  Buy-and-hold baseline:")
        print(f"  Total return:    {bh_total:>8.2%}")
        print(f"  Annualized ret:  {bh_annual:>8.2%}")
        print(f"  Annualized vol:  {bh_vol:>8.2%}")
        print(f"  Sharpe:          {bh_sharpe:>8.3f}")
        print(f"  Max drawdown:    {bh_dd:>8.2%}")