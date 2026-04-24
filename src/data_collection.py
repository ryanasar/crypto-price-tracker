from pathlib import Path

import ccxt
import pandas as pd
import time


def fetch_ohlcv_batch(exchange, symbol, timeframe, since=None, limit=300):
    ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, unit='ms')
    df = df.set_index('timestamp')

    return df


def fetch_ohlcv_range(exchange, symbol, timeframe, start_ms, end_ms, limit=300):
    cursor = start_ms
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    all_batches = []

    while cursor < end_ms:
        batch_df = fetch_ohlcv_batch(
            exchange,
            symbol,
            timeframe,
            since=cursor,
            limit=limit,
        )

        if batch_df.empty:
            break

        all_batches.append(batch_df)

        last_ts = int(batch_df.index[-1].timestamp() * 1000)
        cursor = last_ts + timeframe_ms
        time.sleep(0.25)  # hide from coinbase rate limits

    if not all_batches:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    return pd.concat(all_batches)


def save_ohlcv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(path, engine='pyarrow', compression='snappy')


def validate_ohlcv(df, timeframe):
    report = {
        'n_rows': len(df),
        'start': None,
        'end': None,
        'expected_rows': 0,
        'missing_rows': 0,
        'issues': [],
    }

    # empty
    if len(df) == 0:
        report['issues'].append('EMPTY DataFrame')
        return report

    report['start'] = df.index.min()
    report['end'] = df.index.max()

    # duplicate timestamps
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        report['issues'].append(f'{n_dupes} duplicate timestamps')

    # impossible values
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        report['issues'].append('non-positive prices found')
    if (df['volume'] < 0).any():
        report['issues'].append('negative volume found')
    if (df['high'] < df['low']).any():
        report['issues'].append('high < low on some rows')
    if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
        report['issues'].append('high < open/close on some rows')
    if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
        report['issues'].append('low > open/close on some rows')

    # gaps in timeframe
    freq_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h', '1d': '1D'}
    if timeframe not in freq_map:
        report['issues'].append(f'unsupported timeframe for gap check: {timeframe}')
        return report

    # missing rows
    ideal_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq_map[timeframe],
        tz='UTC',
    )
    missing = ideal_index.difference(df.index)
    report['expected_rows'] = len(ideal_index)
    report['missing_rows'] = len(missing)
    if len(missing) > 0:
        pct = 100 * len(missing) / len(ideal_index)
        report['issues'].append(f'{len(missing)} missing bars ({pct:.2f}% of expected)')

    return report


def load_ohlcv(path):
    df = pd.read_parquet(path, engine='pyarrow')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def collect_all_data(symbols, timeframe, years_back, output_dir, limit=300):
    exchange = ccxt.coinbase()

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(years_back * 365.25 * 24 * 60 * 60 * 1000)

    output_dir = Path(output_dir)

    for symbol in symbols:
        df = fetch_ohlcv_range(
            exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
        )

        filename = f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        path = output_dir / filename
        save_ohlcv(df, path)

        print(f"  Saved {len(df):,} rows to {path}")
        print(f"  Range: {df.index[0]} to {df.index[-1]}")

        report = validate_ohlcv(df, timeframe=timeframe)
        print(f"  Validation: {report['n_rows']:,} rows, {report['missing_rows']} missing of {report['expected_rows']:,}")
        if report['issues']:
            for issue in report['issues']:
                print(issue)


if __name__ == '__main__':
    # standalone run uses the defaults below; main.py drives this via collect_all_data()
    collect_all_data(
        symbols=['BTC/USD', 'ETH/USD'],
        timeframe='1h',
        years_back=4,
        output_dir='data/raw',
    )
