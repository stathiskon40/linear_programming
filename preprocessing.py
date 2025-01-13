import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
from alpaca_trade_api.rest import REST


# #######################################
# # Step 1: Data Download & Preprocessing
# #######################################
API_KEY = "PK0IOM31MI1I30LIHITP"
SECRET_KEY = "XGGF03nNzjeIBYVsNeHfva8odbmRfnUsXpWMwWEj"
BASE_URL = "https://paper-api.alpaca.markets/v2"

api = REST(key_id=API_KEY, secret_key=SECRET_KEY, base_url=BASE_URL,api_version='v2')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define the tickers and time frame
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
timeframe = "1D"  # Daily data
start_date = "2022-01-01"
end_date = "2024-01-01"

# Fetch historical data
data = {}
vwap_data = {}
volume_data = {}

for ticker in tickers:
    barset = api.get_bars(
        ticker, 
        timeframe, 
        start=start_date, 
        end=end_date,
        adjustment='split'
    ).df  # DataFrame
    data[ticker] = barset["close"]
    data[ticker] = data[ticker].transpose()
    vwap_data[ticker] = barset["vwap"]
    volume_data[ticker] = barset["volume"]

# Combine all tickers into one DataFrame
prices_df = pd.DataFrame(data)
prices_df = prices_df.ffill().dropna()  # Forward fill and drop NaNs

# Set the features of the LSTM mode (vwap,volume)
vwap_df = pd.DataFrame(vwap_data).ffill().dropna()
volume_df = pd.DataFrame(volume_data).ffill().dropna()

# Convert DataFrames to numpy arrays
prices = prices_df.values  # shape (T, N)
vwap = vwap_df.values
volume = volume_df.values

returns = prices[1:] / prices[:-1] - 1.0
vwap_returns = vwap[1:] / vwap[:-1] - 1.0

# Ensure volume aligns with other features
volume = volume[1:]


# Calculate 9 EMA and 20 EMA for each stock
ema_9 = prices_df.ewm(span=9, adjust=False).mean()  # 9-period EMA
ema_20 = prices_df.ewm(span=20, adjust=False).mean()  # 20-period EMA

#Calculate returns of EMAs
ema_9_returns = ema_9.pct_change().dropna().values
ema_20_returns = ema_20.pct_change().dropna().values

print(returns.shape, vwap_returns.shape, volume.shape,ema_9_returns.shape,ema_20_returns.shape)

# T = returns.shape[0]
