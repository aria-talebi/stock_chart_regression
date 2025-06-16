import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

def load_data(ticker, period="5d", interval="1m"):
    """
    Load stock data from yfinance and compute common metrics as features.

    Parameters
    ----------
    ticker    : Stock symbol to download data from
    period    : Time period in days, months or years
    interval         : Time interval between observations

    Returns
    -------
    df : Dataframe containing observations and additional metrics
    """
    df = yf.download(ticker, period=period, interval=interval, multi_level_index=False)

    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1)) # Log Return
    df['MA_5']  = df['Close'].rolling(window=5).mean() # Moving Average
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean() # Exponential Moving Average
    df['Vol_5'] = df['LogReturn'].rolling(window=5).std() # Volatility (rolling std of returns)

    # Bollinger Band Width (±2 std around MA)
    df['BollingerWidth_5'] = (
        (df['MA_5'] + 2 * df['Vol_5']) - 
        (df['MA_5'] - 2 * df['Vol_5'])
    )

    # Volume Ratio (current / EMA of volume)
    df['Vol_EMA_10'] = df['Volume'].ewm(span=10, adjust=False).mean()
    df['VolRatio_10'] = df['Volume'] / df['Vol_EMA_10']

    # VWAP (cumulative from session start)
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    # RSI (14-period)
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    return df


def make_dataset(df, features, k, horizon):
    """
    df        : DataFrame with at least your `features` + 'Close'
    features  : list of columns to use as inputs
    k         : window size
    horizon   : minutes ahead

    Returns
    -------
    X : (n_samples, k * len(features)) array
    y : (n_samples,)              array
    """

    # Drop rows where any feature is NaN (so rolling stats align)
    if "Close" not in features:
        df_clean = df[features + ["Close"]].dropna()
    else:
        df_clean = df[features].dropna()

    # Extract aligned arrays
    data   = df_clean[features].values     # shape (N, len(features))
    closes = df_clean["Close"].values      # shape (N,)

    # Build sliding windows
    X, y = [], []
    for i in range(len(data) - k - horizon + 1):
        X.append(data[i : i + k].flatten())
        y.append(closes[i + k + horizon - 1])

    return np.array(X), np.array(y)


def load_full_intraday(symbol: str, interval: str = "1m", chunk_days: int = 7, total_days: int = 29):
    now = datetime.now()
    combined_df = pd.DataFrame()

    for i in range(0, total_days, chunk_days):
        remainder = total_days - i
        offset = chunk_days if remainder >= chunk_days else remainder

        end = now - timedelta(days=i)
        start = end - timedelta(days=offset)

        df = yf.download(
            symbol,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            interval=interval,
            multi_level_index=False,
        )

        combined_df = pd.concat([df, combined_df])

    return combined_df

def load_old_intraday(data_file: str, days: int = 5):
    df = pd.read_csv(data_file, parse_dates=["Datetime"])
    df.set_index("Datetime", inplace=True)
    dates = df.index.normalize().unique()[:days] # Get unique first 5 dates
    df = df[df.index.normalize().isin(dates)] # Mask index by matching one of those dates

    # Create lag features and rolling statistics
    df['ret1'] = df['Close'].pct_change()   # 1-min return
    df['ret2'] = df['Close'].pct_change(periods=2)  # 2-min return
    df['delta1'] = df['Close'].diff()   # absolute diff
    df['delta5'] = df['Close'].diff(periods=5)  # 2-min diff

    window = 10
    df['ma'] = df['Close'].rolling(window).mean()   # moving average
    df['std'] = df['Close'].rolling(window).std()   # rolling volatility
    df['zscore'] = (df['Close'] - df['ma']) / df['std']    # normalized deviation

    df['vol'] = df['Volume']
    df['vol_ema'] = df['vol'].ewm(span=window).mean()    # momentum in volume
    df['vol_ratio'] = df['vol']/df['vol_ema']    # sudden volume spikes
    return df