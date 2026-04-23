import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


# =============================================================================
# SETUP — load data and flatten the MultiIndex yfinance returns
# =============================================================================
df = yf.download('BTC-USD', period='2y', interval='1d', auto_adjust=True)
df.columns = df.columns.get_level_values(0)


# =============================================================================
# SELECTION — .iloc vs .loc
# -----------------------------------------------------------------------------
# .iloc = positional  ("where is it in the list?")    — right-exclusive
# .loc  = label-based ("what is it called?")          — inclusive on both ends
# =============================================================================

df.iloc[:5]                             # first 5 rows, positional
df.loc['2024-01-01':'2024-03-31']       # date range, INCLUSIVE both ends
df['Close'].iloc[-10:]                  # last 10 Close values, positional
df.loc[df.index[-10]:, 'Close']         # same thing, label-based


# =============================================================================
# FILTERING — boolean masks
# -----------------------------------------------------------------------------
# df[mask] keeps rows where the mask is True.
# =============================================================================

high_volume = df[df['Volume'] > df['Volume'].median()]
green_days  = df[df['Close']  > df['Open']]
wide_days   = df[(df['High'] - df['Low']) >= (df['High'] - df['Low']).quantile(0.95)]


# =============================================================================
# CREATING COLUMNS — returns and targets
# -----------------------------------------------------------------------------
# Simple returns are NOT additive across time. Log returns ARE.
# is_up is a classification target (shift(-1) it to predict tomorrow).
# =============================================================================

df['daily_return'] = df['Close'].pct_change()
df['log_return']   = np.log(df['Close'] / df['Close'].shift(1))
df['is_up']        = (df['daily_return'] > 0).astype(int)


# =============================================================================
# ROLLING WINDOWS — moving averages, volatility
# -----------------------------------------------------------------------------
# .rolling(n) = current row + (n-1) prior rows. Past-only, no look-ahead.
# First (n-1) rows are NaN — drop before training.
# =============================================================================

df['ma_14']         = df['Close'].rolling(14).mean()
df['ma_50']         = df['Close'].rolling(50).mean()
df['volatility_14'] = df['daily_return'].rolling(14).std()


# =============================================================================
# RESAMPLING — changing timeframes
# -----------------------------------------------------------------------------
# OHLCV needs DIFFERENT aggregations per column — memorize this dict.
# Frequency strings: 'D' day, 'W' week, 'ME' month-end, 'H' hour, '5min', etc.
# =============================================================================

ohlcv_agg = {
    'Open':   'first',
    'High':   'max',
    'Low':    'min',
    'Close':  'last',
    'Volume': 'sum',
}

weekly  = df.resample('W').agg(ohlcv_agg)
monthly = df.resample('ME').agg(ohlcv_agg)


# =============================================================================
# MERGING — combining assets on a shared DatetimeIndex
# -----------------------------------------------------------------------------
# .join(how='inner') keeps only dates present in BOTH DataFrames.
# Default to 'inner' for cross-asset correlations/features.
# =============================================================================

def load_and_prefix(ticker: str, prefix: str) -> pd.DataFrame:
    """Pull OHLCV, flatten MultiIndex, prefix every column."""
    data = yf.download(ticker, period='2y', interval='1d', auto_adjust=True)
    data.columns = data.columns.get_level_values(0)
    return data.rename(columns={c: f'{prefix}_{c}' for c in data.columns})


btc = load_and_prefix('BTC-USD', 'BTC')
eth = load_and_prefix('ETH-USD', 'ETH')
combined = btc.join(eth, how='inner')

combined['BTC_return'] = combined['BTC_Close'].pct_change()
combined['ETH_return'] = combined['ETH_Close'].pct_change()
rolling_corr = combined['BTC_return'].rolling(30).corr(combined['ETH_return'])


# =============================================================================
# PLOTTING — matplotlib boilerplate for time-series charts
# -----------------------------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Close', color='black', linewidth=1, alpha=0.6)
plt.plot(df.index, df['ma_14'], label='MA 14', color='tab:orange', linewidth=1.5)
plt.plot(df.index, df['ma_50'], label='MA 50', color='tab:blue',   linewidth=1.5)
plt.title('BTC Close with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()


# =============================================================================
# GOTCHAS WORTH REMEMBERING
# -----------------------------------------------------------------------------
# • yfinance returns a MultiIndex — flatten with get_level_values(0) immediately.
# • .loc slices are INCLUSIVE on both ends; .iloc slices are NOT.
# • .rolling() leaves NaN in the first (n-1) rows — drop before training.
# • Use log returns for features, direction (is_up) for classification targets.
# • .shift(-1) on a TARGET = predict tomorrow. .shift(-1) on a FEATURE = leakage.
# • Always default to join(how='inner') for cross-asset data.
# =============================================================================