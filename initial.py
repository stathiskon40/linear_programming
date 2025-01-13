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

factors = np.concatenate([returns, vwap_returns, volume, ema_9_returns, ema_20_returns], axis=1)  # Shape (T, N * 5)


# # Parameters for LSTM
# window_size = 10   # Lookback window
# train_split = 0.8
# num_train = int(T * train_split) - window_size

# X_train_list, y_train_list = [], []
# for i in range(num_train):
#     X_train_list.append(factors[i:i+window_size])
#     y_train_list.append(returns[i+window_size])

# X_train = np.array(X_train_list)  # (num_train, window_size, features)
# y_train = np.array(y_train_list)  # (num_train, N)

# X_test_list, y_test_list = [], []
# for i in range(num_train, T - window_size):
#     X_test_list.append(factors[i:i+window_size])
#     y_test_list.append(returns[i+window_size])

# X_test = np.array(X_test_list)  # (num_test, window_size, features)
# y_test = np.array(y_test_list)  # (num_test, N)

# Convert to torch tensors
# X_train_torch = torch.tensor(X_train, dtype=torch.float32)
# y_train_torch = torch.tensor(y_train, dtype=torch.float32)
# X_test_torch = torch.tensor(X_test, dtype=torch.float32)
# y_test_torch = torch.tensor(y_test, dtype=torch.float32)

# #######################################
# # Step 2: Define LSTM Model for Returns
# #######################################

# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
        
#     def forward(self, x):
#         # x: (batch, window_size, features)
#         out, _ = self.lstm(x)
#         # out: (batch, window_size, hidden_size)
#         out = out[:, -1, :]  # (batch, hidden_size)
#         out = self.fc(out)   # (batch, output_size)
#         return out

# input_size = X_train.shape[2]  # Updated to include all features (returns, volatility, moving avg)
# hidden_size = 64
# output_size = N
# num_layers = 1
# learning_rate = 0.001
# num_epochs = 10000

# model = LSTMModel(input_size, hidden_size, output_size, num_layers)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# model.train()
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     predictions = model(X_train_torch)
#     loss = criterion(predictions, y_train_torch)
#     loss.backward()
#     optimizer.step()
#     if (epoch + 1) % 20 == 0:
#         print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.6f}")

# #########################################
# # Step 3: Evaluate Model and Get a Forecast
# #########################################

# model.eval()
# with torch.no_grad():
#     test_preds = model(X_test_torch)
# test_loss = ((test_preds - y_test_torch)**2).mean().item()
# print(f"Test MSE: {test_loss:.6f}")

# # We'll use the last available test input to get next-day predictions
# last_prediction = test_preds[-1].numpy()  # shape (N,)
# print("Predicted next-day returns:", dict(zip(tickers, last_prediction)))

# ##############################################
# # Step 4: Scenario Generation for LP
# ##############################################
# # Instead of relying on a single predicted return vector,
# # we create multiple scenarios to reflect uncertainty.
# # We'll create scenarios by adding noise around the predicted returns.

# num_scenarios = 5
# # For simplicity, assume we model uncertainty as +/- some perturbation
# # In reality, you might use a stochastic model or historical simulation.
# scenario_returns = []
# base_pred = last_prediction
# for s in range(num_scenarios):
#     # Add some random noise to create scenario s
#     # Let's say up to +/- 0.002 (0.2%) perturbation
#     noise = np.random.normal(0, 0.002, N)
#     scenario = base_pred + noise
#     scenario_returns.append(scenario)

# scenario_returns = np.array(scenario_returns)  # shape (num_scenarios, N)
# print("Scenario returns:\n", scenario_returns)

# ##############################################
# # Step 5: Robust Linear Optimization
# ##############################################
# # We want to pick weights w_i that maximize the minimum return across all scenarios.
# # Introduce a variable alpha representing the minimum return.
# # Maximize alpha subject to:
# # sum_i w_i r_{i,s} >= alpha, for all scenarios s
# # sum_i w_i = 1
# # 0 <= w_i <= 0.3

# from pulp import LpStatus, LpVariable, LpProblem, LpMaximize

# problem = LpProblem("Robust_Portfolio_Optimization", LpMaximize)

# w_vars = [LpVariable(f"w_{i}", lowBound=0, upBound=0.3) for i in range(N)]
# alpha = LpVariable("alpha", lowBound=None)  # can be negative if returns can be negative

# # Objective: Maximize alpha
# problem += alpha, "Maximize_Minimum_Scenario_Return"

# # Constraints:
# # 1) Sum of weights = 1
# problem += lpSum(w_vars) == 1, "Full_Investment"

# # 2) For each scenario, sum_i w_i * r_{i,s} >= alpha
# for s in range(num_scenarios):
#     problem += lpSum([w_vars[i]*scenario_returns[s, i] for i in range(N)]) >= alpha, f"Scenario_{s}_constraint"

# # Solve LP
# problem.solve()

# print("Status:", LpStatus[problem.status])
# optimal_weights = [var.varValue for var in w_vars]
# alpha_value = alpha.varValue

# print("Optimal weights:")
# for tk, wt in zip(tickers, optimal_weights):
#     print(f"{tk}: {wt:.4f}")

# print(f"Minimum guaranteed return across scenarios: {alpha_value:.6f}")

# # Check scenario returns from chosen solution
# scenario_values = []
# for s in range(num_scenarios):
#     val = sum(scenario_returns[s, i]*optimal_weights[i] for i in range(N))
#     scenario_values.append(val)

# print("Scenario values with chosen portfolio:", scenario_values)
# print("Worst-case scenario return:", min(scenario_values))
