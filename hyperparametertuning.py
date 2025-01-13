import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
from alpaca_trade_api.rest import REST
from sklearn.preprocessing import StandardScaler
import optuna  # <-- For automated hyperparameter tuning

############################
# Step 1: Data Preparation
############################
API_KEY = "PK0IOM31MI1I30LIHITP"
SECRET_KEY = "XGGF03nNzjeIBYVsNeHfva8odbmRfnUsXpWMwWEj"
BASE_URL = "https://paper-api.alpaca.markets/v2"

api = REST(key_id=API_KEY, secret_key=SECRET_KEY, base_url=BASE_URL, api_version='v2')

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Define tickers and timeframe
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
timeframe = "4H"
start_date = "2022-01-01"
end_date = "2024-07-03"

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
    ).df
    
    data[ticker] = barset["close"].copy()
    vwap_data[ticker] = barset["vwap"].copy()
    volume_data[ticker] = barset["volume"].copy()

prices_df = pd.DataFrame(data).ffill().dropna()
vwap_df = pd.DataFrame(vwap_data).ffill().dropna()
volume_df = pd.DataFrame(volume_data).ffill().dropna()

# Ensure all data is aligned by date/time
common_index = prices_df.index.intersection(vwap_df.index).intersection(volume_df.index)
prices_df = prices_df.loc[common_index]
vwap_df = vwap_df.loc[common_index]
volume_df = volume_df.loc[common_index]

prices = prices_df.values  # Shape: (T, N)
vwap = vwap_df.values      # Shape: (T, N)
volume = volume_df.values  # Shape: (T, N)

# Compute returns
returns = prices[1:] / prices[:-1] - 1.0      # (T-1, N)
vwap_returns = vwap[1:] / vwap[:-1] - 1.0     # (T-1, N)
volume = volume[1:]                           # (T-1, N)

# Calculate EMAs and their returns
ema_9 = prices_df.ewm(span=9, adjust=False).mean()
ema_20 = prices_df.ewm(span=20, adjust=False).mean()

ema_9_returns = ema_9.pct_change().dropna().values    # (T-1, N)
ema_20_returns = ema_20.pct_change().dropna().values  # (T-1, N)

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

# Compute indicators
macd_line, macd_signal, macd_hist = compute_macd(prices_df)
rsi_df = compute_rsi(prices_df)
middle_band, upper_band, lower_band = compute_bollinger_bands(prices_df)

# Align all to the same length
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

# Truncate arrays to `min_length`
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

macd_line_arr = macd_line.values       # (min_length, N)
rsi_arr = rsi_df.values                # (min_length, N)
middle_band_arr = middle_band.values   # (min_length, N)
upper_band_arr = upper_band.values     # (min_length, N)
lower_band_arr = lower_band.values     # (min_length, N)

# Combine into one factor matrix
factors = np.concatenate([
    returns, vwap_returns, volume,
    ema_9_returns, ema_20_returns,
    macd_line_arr, rsi_arr,
    upper_band_arr, lower_band_arr,
    middle_band_arr
], axis=1)

print("Factors shape:", factors.shape)

# Prepare data for model
seq_length = 30
T = factors.shape[0]
N = returns.shape[1]

train_ratio = 0.8
train_size = int(T * train_ratio)
test_size = T - train_size

factors_train = factors[:train_size]
returns_train = returns[:train_size]

factors_test = factors[train_size:]
returns_test = returns[train_size:]

# We'll apply scaling just once, outside the objective function.
scaler = StandardScaler()
scaler.fit(factors_train)  
factors_train_scaled = scaler.transform(factors_train)
factors_test_scaled  = scaler.transform(factors_test)

# Dataset & Dataloader
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

# We only create the "raw" dataset objects once
# (We'll re-instantiate them as needed inside objective with trial-suggested batch size)
train_dataset_whole = TimeSeriesDataset(factors_train_scaled, returns_train, seq_length=seq_length)
test_dataset_whole  = TimeSeriesDataset(factors_test_scaled,  returns_test,  seq_length=seq_length)

print("Train dataset length:", len(train_dataset_whole))
print("Test dataset length:", len(test_dataset_whole))

########################################
# LSTM Model Class (same as original)
########################################
class FinancialLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(FinancialLSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output

device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################################
# Optuna Objective Function
########################################
def objective(trial):
    """
    Each trial will:
      1. Suggest hyperparameters.
      2. Create new DataLoaders with the suggested batch_size.
      3. Initialize and train the model using the suggested hyperparams.
      4. Return the validation/test loss.
    """

    # ----------- 1) Suggest Hyperparams -----------
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout    = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # ----------- 2) DataLoaders with new batch_size -----------
    train_loader = DataLoader(train_dataset_whole, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset_whole,  batch_size=batch_size, shuffle=False)

    # ----------- 3) Create & Train Model -----------
    input_dim = factors.shape[1]  # features per timestep
    output_dim = N                # number of stocks (next-day returns)
    model = FinancialLSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    EPOCHS = 20  # You can also tune epochs if desired
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # Training Loop
        model.train()
        running_train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * X_batch.size(0)
        
        # Average train loss for the epoch
        epoch_train_loss = running_train_loss / len(train_dataset_whole)
        
        # Validation/Test Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                val_preds = model(X_val)
                batch_loss = criterion(val_preds, y_val)
                val_loss += batch_loss.item() * X_val.size(0)
        val_loss /= len(test_dataset_whole)
        
        # If you'd like to see intermediate results during the tune:
        # print(f"Epoch {epoch+1}/{EPOCHS}: train_loss={epoch_train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        # Optionally early-stop if no improvement, etc.

    return best_val_loss

########################################
# Step: Run the Optuna Study
########################################
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    # Number of trials can be changed based on your resources/time
    study.optimize(objective, n_trials=20)

    print("\n=======================")
    print("Number of finished trials:", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value (test loss):", trial.value)
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Once the study is done, you can retrieve the best hyperparams,
    # re-instantiate and retrain a final model, then do predictions:
    best_params = study.best_params
    print("\nBest hyperparameters found by Optuna:", best_params)

    # Example: retrain final model on the entire training set with best params
    best_hidden_dim = best_params["hidden_dim"]
    best_num_layers = best_params["num_layers"]
    best_dropout    = best_params["dropout"]
    best_lr         = best_params["lr"]
    best_batch_size = best_params["batch_size"]

    # Re-create data loaders with best batch_size
    final_train_loader = DataLoader(train_dataset_whole, batch_size=best_batch_size, shuffle=True)
    final_test_loader  = DataLoader(test_dataset_whole,  batch_size=best_batch_size, shuffle=False)

    final_model = FinancialLSTMModel(
        input_dim=factors.shape[1],
        hidden_dim=best_hidden_dim,
        num_layers=best_num_layers,
        output_dim=N,
        dropout=best_dropout
    ).to(device)

    final_optimizer = optim.Adam(final_model.parameters(), lr=best_lr)
    final_criterion = nn.MSELoss()

    # Example final training loop (same approach, possibly more epochs)
    FINAL_EPOCHS = 50
    for epoch in range(FINAL_EPOCHS):
        final_model.train()
        total_loss = 0.0
        for X_batch, y_batch in final_train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            final_optimizer.zero_grad()
            preds = final_model(X_batch)
            loss = final_criterion(preds, y_batch)
            loss.backward()
            final_optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
        avg_train_loss = total_loss / len(train_dataset_whole)

        final_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in final_test_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                val_preds = final_model(X_val)
                val_loss += final_criterion(val_preds, y_val).item() * X_val.size(0)
        avg_val_loss = val_loss / len(test_dataset_whole)

        print(f"[Final Model] Epoch {epoch+1}/{FINAL_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_val_loss:.4f}")

    # Once final model is trained, do your usual "next-day returns" inference if desired:
    if len(test_dataset_whole) > 0:
        X_last, y_last = test_dataset_whole[len(test_dataset_whole)-1]
        if X_last.shape[0] != seq_length:
            raise ValueError(f"Expected X_last to have sequence length {seq_length}, got {X_last.shape[0]}")

        final_model.eval()
        X_last = X_last.unsqueeze(0).to(device)
        with torch.no_grad():
            final_preds = final_model(X_last).cpu().numpy()[0]

        print("\n----- Next-Day Returns Comparison (Using Final Model) -----")
        for i, ticker in enumerate(tickers):
            print(f"{ticker}: Predicted = {final_preds[i]*100:.6f}%, Actual = {y_last.numpy()[i]*100:.6f}%")

