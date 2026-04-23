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
        time.sleep(0.25) # hide from coinbase rate limits
    
    if not all_batches:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    return pd.concat(all_batches)

if __name__ == '__main__':
    exchange = ccxt.coinbase()
    
    end_ms = int(time.time() * 1000) # now in ms
    start_ms = end_ms - (7 * 24 * 60 * 60 * 1000)  # 7 days ago in ms
    
    df = fetch_ohlcv_range(
        exchange,
        symbol='BTC/USD',
        timeframe='1h',
        start_ms=start_ms,
        end_ms=end_ms,
        limit=100,
    )
    