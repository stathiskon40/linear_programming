import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
from alpaca_trade_api.rest import REST
############################
# Step 1: Data Preparation
############################
API_KEY = "PK0IOM31MI1I30LIHITP"
SECRET_KEY = "XGGF03nNzjeIBYVsNeHfva8odbmRfnUsXpWMwWEj"
BASE_URL = "https://paper-api.alpaca.markets/v2"

api = REST(key_id=API_KEY, secret_key=SECRET_KEY, base_url=BASE_URL,api_version='v2')

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Define tickers and timeframe
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
timeframe = "1D"
start_date = "2018-01-01"
# end_date = "2024-01-01"
# end_date = datetime.now().strftime('%Y-%m-%d')  # Today's date
end_date = "2024-06-01"

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

prices = prices_df.values  # Shape: (T, N)
vwap = vwap_df.values
volume = volume_df.values

# Compute returns
returns = prices[1:] / prices[:-1] - 1.0            # (T-1, N)
vwap_returns = vwap[1:] / vwap[:-1] - 1.0           # (T-1, N)
volume = volume[1:]                                 # (T-1, N)

# Calculate EMAs and their returns
ema_9 = prices_df.ewm(span=9, adjust=False).mean()
ema_20 = prices_df.ewm(span=20, adjust=False).mean()

ema_9_returns = ema_9.pct_change().dropna().values   # (T-1, N)
ema_20_returns = ema_20.pct_change().dropna().values # (T-1, N)

# Ensure all have the same shape (T-1, N)
# Check shapes: 
# returns.shape, vwap_returns.shape, volume.shape, ema_9_returns.shape, ema_20_returns.shape
print("Shapes:", returns.shape, vwap_returns.shape, volume.shape, ema_9_returns.shape, ema_20_returns.shape)

# Combine all features into one factor matrix: (T-1, N*5)
factors = np.concatenate([returns, vwap_returns, volume, ema_9_returns, ema_20_returns], axis=1)
print("Factors shape:", factors.shape)

########################################
# Step 2: Preparing the LSTM Training
########################################

# For simplicity, let's predict the next day's returns for all N stocks using the last 30 days.
seq_length = 30
T = factors.shape[0]
N = returns.shape[1]

# Targets: next-day returns
# If we use returns directly as targets, we need to ensure the dataset sequences align.
# Our input sequences will be from factors[t-30:t], and we predict returns[t].
# Because we lose 30 steps at the start, the maximum index for t is from seq_length to T-1.

# Train/test split (time-based)
train_ratio = 0.8
train_size = int(T * train_ratio)
test_size = T - train_size

factors_train = factors[:train_size]
returns_train = returns[:train_size]

factors_test = factors[train_size:]
returns_test = returns[train_size:]

class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, seq_length=30):
        # data: (num_samples, input_dim)
        # targets: (num_samples, output_dim)
        self.data = data
        self.targets = targets
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        X = self.data[idx:idx+self.seq_length]        # (seq_length, input_dim)
        y = self.targets[idx+self.seq_length]         # prediction for day after the sequence
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_dataset = TimeSeriesDataset(factors_train, returns_train, seq_length=seq_length)
test_dataset = TimeSeriesDataset(factors_test, returns_test, seq_length=seq_length)
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
        # x: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the last time step output
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output

# Initialize model, criterion, optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_dim = factors.shape[1]   # N*5
hidden_dim = 64
num_layers = 2
output_dim = N   # predict returns for each stock
model = FinancialLSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

########################################
# Step 4: Training the Model
########################################

epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss = train_loss / len(train_dataset)
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in test_loader:
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val)
            test_loss += val_loss.item() * X_val.size(0)
    test_loss = test_loss / len(test_dataset)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

########################################
# Step 5: Using the Model
########################################

# Example: Predict on the last sequence of the test set
# X_last, y_last = test_dataset[-1]
X_last, y_last = test_dataset[len(test_dataset)-1]
X_last = X_last.unsqueeze(0).to(device)
model.eval()
predicted_returns = model(X_last).detach().cpu().numpy()
print("Predicted next-day returns for each stock:", predicted_returns)
