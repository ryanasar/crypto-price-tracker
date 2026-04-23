def simulate_strategy(df, upper_threshold=0.55, lower_threshold=0.45, fee_per_side=0.002):
    df['return'] = df['close'].pct_change()

    position = 0  # start in cash
    positions = []
    for prob in df['y_prob']:
        if prob > upper_threshold:
            position = 1
        elif prob < lower_threshold:
            position = 0
        # else: hold current position
        positions.append(position)
    df['position'] = positions

    df['trade'] = (df['position'] != df['position'].shift(1)).astype(int)
    df.loc[df.index[0], 'trade'] = 0   # first row can't be a trade

    # shift position forward, position at t applies to return at t+1
    df['pnl'] = df['position'].shift(1) * df['return']

    # remove fee
    df['pnl'] -= df['trade'].shift(1) * fee_per_side

    df['equity'] = (1 + df['pnl'].fillna(0)).cumprod()

    return df


def compute_metrics(result, periods_per_year=24 * 365):
    total_return = result['equity'].iloc[-1] - 1
    
    n_bars = len(result)
    years_elapsed = n_bars / periods_per_year
    
    if years_elapsed > 0 and (1 + total_return) > 0:
        annualized_return = (1 + total_return) ** (1 / years_elapsed) - 1
    else:
        annualized_return = 0
    
    annualized_vol = result['pnl'].std() * (periods_per_year ** 0.5)
    
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    running_max = result['equity'].expanding().max()
    drawdown_series = result['equity'] / running_max - 1
    max_drawdown = drawdown_series.min()
    
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    num_trades = int(result['trade'].sum())
    fraction_long = result['position'].mean()
    
    in_market_mask = result['position'].shift(1) == 1
    in_market_pnl = result['pnl'][in_market_mask]
    
    if len(in_market_pnl) > 0:
        wins = in_market_pnl[in_market_pnl > 0]
        losses = in_market_pnl[in_market_pnl < 0]
        win_rate = len(wins) / len(in_market_pnl)
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    else:
        win_rate = avg_win = avg_loss = expectancy = 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'num_trades': num_trades,
        'fraction_long': fraction_long,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
    }
