import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
from alpaca_trade_api.rest import REST
from sklearn.preprocessing import StandardScaler  # <-- NEW
from tabulate import tabulate
import yfinance as yf


from basic_lp import DataProcessor
from lp_diversification_constraints import optimize_portfolio
from lp_with_risk_management import optimize_portfolio_with_risk

import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna

import os 

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

############################
# Step 1: Data Preparation
############################

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# Define tickers and timeframe
# tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
tickers = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "AMZN",  # Amazon.com Inc.
    "TSLA",  # Tesla Inc.
    "GOOGL", # Alphabet Inc.
    "META",  # Meta Platforms Inc.
    "NVDA",  # NVIDIA Corporation
    "BRK-B", # Berkshire Hathaway Inc.
    "JNJ",   # Johnson & Johnson
    "UNH",   # UnitedHealth Group Incorporated
    "XOM",   # Exxon Mobil Corporation
    "V",     # Visa Inc.
    "PG",    # Procter & Gamble Co.
    "JPM",   # JPMorgan Chase & Co.
    "DIS",   # The Walt Disney Company
    "KO",    # Coca-Cola Company
    "PEP",   # PepsiCo Inc.
    "COST",  # Costco Wholesale Corporation
    "NFLX",  # Netflix Inc.
    "PFE"    # Pfizer Inc.
]

timeframe = "1D"
start_date = "2000-01-01"
end_date = "2024-09-26"


data = None
vwap_data = None
volume_data = None


# Define indicator functions
def compute_macd(prices, short_span=12, long_span=26, signal_span=9):
    ema_short = prices.ewm(span=short_span, adjust=False).mean()
    ema_long = prices.ewm(span=long_span, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(prices, window=20, num_std=2):
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return sma, upper_band, lower_band


file_path = 'data.csv'
data_already_exist = False
data_wrong =  False 
factors = None
returns = None

if os.path.exists(file_path):
    data_already_exist = True
    stored_data = pd.read_csv(file_path)
    tickers = stored_data['tickers'].dropna().tolist()
    stored_data.drop(columns=['tickers'], inplace=True)
    
    if set(tickers) != set(tickers):
        data_wrong = True
        print("Data is wrong")
    
    if data_already_exist == True and data_wrong == False: 
        print("Data already exists. Fetching data from file.")
        factors =  stored_data.values
        
        column_indices = [i for i in range(0, factors.shape[1], 10)]  # 0-based indexing
        returns = stored_data.iloc[:, column_indices].to_numpy()


if (data_already_exist ==  False) or (data_wrong == True):
    
    data = pd.DataFrame()
    vwap_data = pd.DataFrame()
    volume_data = pd.DataFrame()
    
    for ticker in tickers:
        barset = yf.download(ticker, start=start_date, end=end_date, progress=True)
        barset = pd.DataFrame(barset)
        
        data[ticker] = barset["Close"]
        volume_data[ticker] = barset["Volume"].copy()
        vwap_data[ticker] = (data[ticker] * volume_data[ticker]).cumsum() / volume_data[ticker].cumsum()

    data.reset_index(drop=True, inplace=True)
    vwap_data.reset_index(drop=True, inplace=True)
    volume_data.reset_index(drop=True, inplace=True)

    prices_df = data.ffill().dropna()
    vwap_df = vwap_data.ffill().dropna()
    volume_df = volume_data.ffill().dropna()

    common_index = prices_df.index.intersection(vwap_df.index).intersection(volume_df.index)
    prices_df = prices_df.loc[common_index]
    vwap_df = vwap_df.loc[common_index]
    volume_df = volume_df.loc[common_index]

    prices = prices_df.values  
    vwap = vwap_df.values
    volume = volume_df.values

    # Compute returns
    returns = prices[1:] / prices[:-1] - 1.0
    vwap_returns = vwap[1:] / vwap[:-1] - 1.0
    volume = volume[1:]

    # Calculate EMAs and their returns
    ema_9 = prices_df.ewm(span=9, adjust=False).mean()
    ema_20 = prices_df.ewm(span=20, adjust=False).mean()

    ema_9_returns = ema_9.pct_change().dropna().values
    ema_20_returns = ema_20.pct_change().dropna().values

    # Compute indicators
    macd_line, macd_signal, macd_hist = compute_macd(prices_df)
    rsi_df = compute_rsi(prices_df)
    middle_band, upper_band, lower_band = compute_bollinger_bands(prices_df)

    # Drop NaNs and align
    min_length = min(
        len(prices_df)-1,
        len(macd_line.dropna()),
        len(rsi_df.dropna()),
        len(middle_band.dropna()),
        len(upper_band.dropna()),
        len(lower_band.dropna()),
        returns.shape[0],
        vwap_returns.shape[0],
        volume.shape[0],
        ema_9_returns.shape[0],
        ema_20_returns.shape[0]
    )

    returns = returns[-min_length:]
    vwap_returns = vwap_returns[-min_length:]
    volume = volume[-min_length:]
    ema_9_returns = ema_9_returns[-min_length:]
    ema_20_returns = ema_20_returns[-min_length:]

    macd_line = macd_line.dropna().iloc[-min_length:]
    rsi_df = rsi_df.dropna().iloc[-min_length:]
    middle_band = middle_band.dropna().iloc[-min_length:]
    upper_band = upper_band.dropna().iloc[-min_length:]
    lower_band = lower_band.dropna().iloc[-min_length:]

    macd_line_arr = macd_line.values
    rsi_arr = rsi_df.values
    middle_band_arr = middle_band.values
    upper_band_arr = upper_band.values
    lower_band_arr = lower_band.values

    # Original factors
    factors = np.concatenate([
        returns, vwap_returns, volume, 
        ema_9_returns, ema_20_returns,
        macd_line_arr, rsi_arr, 
        upper_band_arr, lower_band_arr, 
        middle_band_arr
    ], axis=1)
    print("Factors shape:", factors.shape)
    
    # Gerenate column names
    column_names = [f"Column_{i+1}" if (i % 10 == 0) else "" for i in range(factors.shape[1])]
    
    stored_data = pd.DataFrame(factors,columns=column_names)
    print(stored_data.shape[0])
    tickers_new_list = tickers + [np.nan] * (stored_data.shape[0] - len(tickers))
    stored_data['tickers'] = tickers_new_list
    # stored_data.to_csv(file_path,index=False)
    
########################################
# Step 2: Preparing the LSTM Training
########################################

seq_length = 30
T = factors.shape[0]
N = returns.shape[1]
# N =  len(tickers)

train_ratio = 0.8
train_size = int(T * train_ratio)
test_size = T - train_size

factors_train = factors[:train_size]
returns_train = returns[:train_size]

factors_test = factors[train_size:]
returns_test = returns[train_size:]

# -------- NEW LINES FOR NORMALIZATION --------
scaler = StandardScaler()
scaler.fit(factors_train)                  # Only fit on training set
factors_train_scaled = scaler.transform(factors_train)
factors_test_scaled  = scaler.transform(factors_test)
# ---------------------------------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, seq_length=30):
        self.data = data
        self.targets = targets
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        X = self.data[idx:idx+self.seq_length]
        y = self.targets[idx+self.seq_length]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Use the scaled datasets now
train_dataset = TimeSeriesDataset(factors_train_scaled, returns_train, seq_length=seq_length)
test_dataset = TimeSeriesDataset(factors_test_scaled, returns_test, seq_length=seq_length)

print("Train dataset length:", len(train_dataset))
print("Test dataset length:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

########################################
# Step 3: Defining the Model
########################################

class FinancialLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(FinancialLSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output



device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_dim = factors.shape[1]
# hidden_dim = 64
# num_layers = 4
output_dim = N


# Assuming FinancialLSTMModel, train_loader, test_loader, device, input_dim, output_dim, etc. are defined above

# def train_and_evaluate_model(hidden_dim, num_layers, learning_rate, epochs=20,dropout_optuna=0.2):
#     # Initialize the model with given hyperparameters
#     model = FinancialLSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout_optuna).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
#     # Training loop
#     model.train()
#     for epoch in range(epochs):
#         for X_batch, y_batch in train_loader:
#             X_batch = X_batch.to(device)
#             y_batch = y_batch.to(device)

#             optimizer.zero_grad()
#             predictions = model(X_batch)
#             loss = criterion(predictions, y_batch)
#             loss.backward()
#             optimizer.step()
    
#     # Evaluate directional accuracy on the test set
#     model.eval()
#     all_predictions = []
#     all_actuals = []

#     with torch.no_grad():
#         for X_batch, y_batch in test_loader:
#             X_batch = X_batch.to(device)
#             predictions = model(X_batch).cpu().numpy()
#             all_predictions.append(predictions)
#             all_actuals.append(y_batch.numpy())

#     all_predictions = np.concatenate(all_predictions, axis=0)
#     all_actuals = np.concatenate(all_actuals, axis=0)

#     predicted_direction = np.sign(all_predictions)
#     actual_direction = np.sign(all_actuals)
#     correct_direction = (predicted_direction == actual_direction)
#     directional_accuracy = np.mean(correct_direction) * 100

#     return directional_accuracy

# def train_and_evaluate_model(hidden_dim, num_layers, learning_rate, epochs=20,dropout_optuna=0.2):
#     # Initialize model with given hyperparameters
#     model = FinancialLSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout_optuna).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
#     # Training loop
#     for epoch in range(epochs):
#         model.train()
#         for X_batch, y_batch in train_loader:
#             X_batch = X_batch.to(device)
#             y_batch = y_batch.to(device)

#             optimizer.zero_grad()
#             predictions = model(X_batch)
#             loss = criterion(predictions, y_batch)
#             loss.backward()
#             optimizer.step()
    
#     # Evaluate directional accuracy on the test set
#     model.eval()
#     all_predictions = []
#     all_actuals = []

#     with torch.no_grad():
#         for X_batch, y_batch in test_loader:
#             X_batch = X_batch.to(device)
#             predictions = model(X_batch).cpu().numpy()
#             all_predictions.append(predictions)
#             all_actuals.append(y_batch.numpy())

#     all_predictions = np.concatenate(all_predictions, axis=0)
#     all_actuals = np.concatenate(all_actuals, axis=0)

#     predicted_direction = np.sign(all_predictions)
#     actual_direction = np.sign(all_actuals)
#     correct_direction = (predicted_direction == actual_direction)
#     directional_accuracy = np.mean(correct_direction) * 100

#     return directional_accuracy

def objective(trial):
    # Suggest hyperparameters to optimize
    # hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    # num_layers = trial.suggest_int('num_layers', 2, 4)
    # learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 8)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 5e-3)
    dropout = trial.suggest_categorical('dropout', [0.0, 0.2, 0.5])
    
    # Train model and return directional accuracy
    accuracy = train_and_evaluate_model(hidden_dim, num_layers, learning_rate, epochs=20,dropout_optuna=dropout)
    return accuracy

# Create a study object with direction "maximize" since we want to maximize directional accuracy
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100,show_progress_bar=True)

print("Best trial:")
trial = study.best_trial
print(f"Params: {trial.params}")
print(f"Directional Accuracy: {trial.value:.2f}%")