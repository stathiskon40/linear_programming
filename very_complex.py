import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
from alpaca_trade_api.rest import REST
from concurrent.futures import ThreadPoolExecutor
import pandas_datareader.data as web
import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define all functions and classes here
def fetch_ticker_data(ticker, api, timeframe, start_date, end_date):
    try:
        barset = api.get_bars(
            ticker, 
            timeframe, 
            start=start_date, 
            end=end_date,
            adjustment='split'
        ).df
        logging.info(f"Fetched data for {ticker}")
        return ticker, barset
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return ticker, pd.DataFrame()  # Return empty DataFrame on error

def compute_obv(volume_df, close_df):
    obv = (np.sign(close_df.diff()) * volume_df).fillna(0).cumsum()
    return obv

def compute_macd(prices_df, short_span=12, long_span=26, signal_span=9):
    ema_short = prices_df.ewm(span=short_span, adjust=False).mean()
    ema_long = prices_df.ewm(span=long_span, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def compute_rsi(prices_df, period=14):
    delta = prices_df.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window=period, min_periods=period).mean()
    loss = down.rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(prices_df, window=20, num_std=2):
    sma = prices_df.rolling(window).mean()
    std = prices_df.rolling(window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return sma, upper_band, lower_band

def compute_atr(high_df, low_df, close_df, window=14):
    tr1 = high_df - low_df
    tr2 = (high_df - close_df.shift()).abs()
    tr3 = (low_df - close_df.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def compute_stochastic_oscillator(prices_df, window=14):
    low_min = prices_df.rolling(window=window).min()
    high_max = prices_df.rolling(window=window).max()
    stochastic = 100 * (prices_df - low_min) / (high_max - low_min)
    return stochastic

class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, seq_length=30):
        self.data = data
        self.targets = targets
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        X = self.data[idx:idx+self.seq_length]        # (seq_length, input_dim)
        y = self.targets[idx+self.seq_length]         # prediction for day after the sequence
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)  # For bidirectional

    def forward(self, lstm_output):
        scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        context = torch.sum(weights * lstm_output, dim=1)  # (batch, hidden_dim*2)
        return context

class EnhancedFinancialLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(EnhancedFinancialLSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        
        self.attention = Attention(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        return out

def main():
    # Define your API credentials and other parameters
    API_KEY = "PK0IOM31MI1I30LIHITP"
    SECRET_KEY = "XGGF03nNzjeIBYVsNeHfva8odbmRfnUsXpWMwWEj"
    BASE_URL = "https://paper-api.alpaca.markets/v2"
    
    api = REST(key_id=API_KEY, secret_key=SECRET_KEY, base_url=BASE_URL, api_version='v2')
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Define tickers and timeframe
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    timeframe = "1D"
    start_date = "2022-01-01"
    end_date = "2024-07-03"
    
    data = {}
    vwap_data = {}
    volume_data = {}
    high_data = {}
    low_data = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_ticker_data, tickers, [api]*len(tickers), [timeframe]*len(tickers), [start_date]*len(tickers), [end_date]*len(tickers))
    
    for ticker, barset in results:
        if not barset.empty:
            data[ticker] = barset["close"].copy()
            vwap_data[ticker] = barset["vwap"].copy()
            volume_data[ticker] = barset["volume"].copy()
            high_data[ticker] = barset["high"].copy()
            low_data[ticker] = barset["low"].copy()
        else:
            logging.warning(f"No data for {ticker}, skipping.")
    
    # Create DataFrames and handle missing data
    prices_df = pd.DataFrame(data).ffill().dropna()
    vwap_df = pd.DataFrame(vwap_data).ffill().dropna()
    volume_df = pd.DataFrame(volume_data).ffill().dropna()
    high_df = pd.DataFrame(high_data).ffill().dropna()
    low_df = pd.DataFrame(low_data).ffill().dropna()
    
    # Fetch macroeconomic data (e.g., GDP)
    try:
        logging.info("Fetching GDP data...")
        gdp = web.DataReader('GDP', 'fred', start_date, end_date).ffill()
        # Remove timezone from gdp index if present
        if gdp.index.tz is not None:
            gdp.index = gdp.index.tz_convert(None)
        # Reindex gdp to match prices_df's index with forward and backward fill
        gdp = gdp.reindex(prices_df.index).ffill().bfill()
        logging.info("GDP data aligned with prices_df")
    except Exception as e:
        logging.error(f"Error fetching or processing GDP data: {e}")
        # Handle the error, e.g., exit or proceed with caution
        gdp = pd.Series(index=prices_df.index, data=np.nan)
        gdp = gdp.ffill().bfill()
    
    # Compute returns and other financial indicators
    returns = prices_df.pct_change().dropna().values            # (T, N)
    vwap_returns = vwap_df.pct_change().dropna().values        # (T, N)
    volume = volume_df.values[1:]                              # (T-1, N)
    high = high_df.values[1:]
    low = low_df.values[1:]
    prices = prices_df.values[1:]
    gdp_values = gdp.values[1:]                                # (T-1, 1)
    
    # Calculate EMAs and their returns
    ema_9 = prices_df.ewm(span=9, adjust=False).mean().pct_change().dropna().values   # (T-2, N)
    ema_20 = prices_df.ewm(span=20, adjust=False).mean().pct_change().dropna().values # (T-2, N)
    
    # Compute indicators
    macd_line, macd_signal, macd_hist = compute_macd(prices_df)
    rsi_df = compute_rsi(prices_df)
    middle_band, upper_band, lower_band = compute_bollinger_bands(prices_df)
    atr = compute_atr(high_df, low_df, prices_df['AAPL'])  # Example for AAPL
    stochastic = compute_stochastic_oscillator(prices_df)
    obv = compute_obv(volume_df, prices_df)
    
    # Combine indicators into a single DataFrame
    indicators = pd.concat([
        macd_line, macd_signal, macd_hist,
        rsi_df, middle_band, upper_band, lower_band,
        atr, stochastic, obv
    ], axis=1).dropna()
    
    # Align all data to the minimum length
    min_length = min(len(returns), len(vwap_returns), len(volume),
                    len(ema_9), len(ema_20), len(indicators), len(gdp_values))
    
    returns = returns[-min_length:]
    vwap_returns = vwap_returns[-min_length:]
    volume = volume[-min_length:]
    high = high[-min_length:]
    low = low[-min_length:]
    prices = prices[-min_length:]
    gdp_values = gdp_values[-min_length:]
    
    ema_9 = ema_9[-min_length:]
    ema_20 = ema_20[-min_length:]
    indicators = indicators.iloc[-min_length:].values
    
    # Original factors: returns, vwap_returns, volume, ema_9_returns, ema_20_returns, indicators, macro data
    factors = np.concatenate([
        returns, 
        vwap_returns, 
        volume, 
        ema_9, 
        ema_20,
        indicators,
        gdp_values
    ], axis=1)
    
    # Feature Scaling
    scaler = StandardScaler()
    factors_scaled = scaler.fit_transform(factors)
    
    logging.info(f"Factors shape after scaling: {factors_scaled.shape}")
    
    # Check for NaN or Inf values
    if np.isnan(factors_scaled).any() or np.isinf(factors_scaled).any():
        logging.error("NaN or Inf values found in factors_scaled. Replacing with zeros.")
        factors_scaled = np.nan_to_num(factors_scaled)
    
    # Define targets: next-day returns
    targets = returns  # For regression; adjust accordingly if using classification
    
    # Create datasets
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2
    
    T = factors_scaled.shape[0]
    train_size = int(T * train_ratio)
    val_size = int(T * val_ratio)
    
    factors_train = factors_scaled[:train_size]
    targets_train = targets[:train_size]
    
    factors_val = factors_scaled[train_size:train_size+val_size]
    targets_val = targets[train_size:train_size+val_size]
    
    factors_test = factors_scaled[train_size+val_size:]
    targets_test = targets[train_size+val_size:]
    
    train_dataset = TimeSeriesDataset(factors_train, targets_train, seq_length=30)
    val_dataset = TimeSeriesDataset(factors_val, targets_val, seq_length=30)
    test_dataset = TimeSeriesDataset(factors_test, targets_test, seq_length=30)
    
    logging.info(f"Train dataset length: {len(train_dataset)}")
    logging.info(f"Validation dataset length: {len(val_dataset)}")
    logging.info(f"Test dataset length: {len(test_dataset)}")
    
    # Initialize DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Define the model
    input_dim = factors_scaled.shape[1]   # Enhanced input dim with additional indicators
    hidden_dim = 128                       # Increased hidden dimensions
    num_layers = 3                         # More layers for deeper learning
    output_dim = N                         # Predict returns for each stock
    model = EnhancedFinancialLSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
    
    # Define optimizer, scheduler, and scaler
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    # Initialize GradScaler conditionally based on CUDA availability
    if torch.cuda.is_available():
        scaler = torch.amp.GradScaler()
    else:
        scaler = torch.amp.GradScaler(enabled=False)
        logging.warning("CUDA is not available. GradScaler disabled.")
    
    # Training loop
    epochs = 100
    best_loss = float('inf')
    patience = 10
    counter = 0
    checkpoint_path = 'checkpoints'
    os.makedirs(checkpoint_path, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    val_preds = model(X_val)
                    loss = criterion(val_preds, y_val)
                val_loss += loss.item() * X_val.size(0)
        val_loss /= len(val_dataset)
        
        scheduler.step(val_loss)
        
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early Stopping and Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'best_model.pth'))
            logging.info("Best model saved.")
        else:
            counter += 1
            if counter >= patience:
                logging.info("Early stopping triggered.")
                break
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'best_model.pth')))
    
    # Evaluation on Test Set
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device, non_blocking=True)
            y_test = y_test.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                preds = model(X_test)
                loss = criterion(preds, y_test)
            test_loss += loss.item() * X_test.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_test.cpu().numpy())
    test_loss /= len(test_dataset)
    logging.info(f"Test Loss: {test_loss:.6f}")
    
    # Compute additional metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    logging.info(f"Test MAE: {mae:.6f}, Test R²: {r2:.6f}")
    
    # Predicting the next day returns
    if len(test_dataset) > 0:
        X_last, y_last = test_dataset[-1]
        X_last = X_last.unsqueeze(0).to(device, non_blocking=True)
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                predicted_returns = model(X_last).cpu().numpy()
        for i in range(len(tickers)):
            print(f"{tickers[i]}: {predicted_returns[0][i] * 100:.6f}%")

if __name__ == "__main__":
    main()
