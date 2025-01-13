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

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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


############################
# Step 1: Data Preparation
############################

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

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
end_date = "2024-11-27"


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
hidden_dim = 32
num_layers = 7
output_dim = N

# model = FinancialLSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.00026704844160296224)


global model 
model = None
 
########################################
# Step 4: Training the Model
########################################
def train_and_evaluate_model(hidden_dim, num_layers, learning_rate, epochs=20):
    # Initialize model with given hyperparameters
    global model  
    model = FinancialLSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout=0.5).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())


        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
         
        # Evaluate directional accuracy on the test set
        model.eval()
        all_predictions = []
        all_actuals = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                predictions = model(X_batch).cpu().numpy()
                all_predictions.append(predictions)
                all_actuals.append(y_batch.numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_actuals = np.concatenate(all_actuals, axis=0)

        predicted_direction = np.sign(all_predictions)
        actual_direction = np.sign(all_actuals)
        correct_direction = (predicted_direction == actual_direction)
        directional_accuracy = np.mean(correct_direction) * 100
        
        print(f"Epoch {epoch}/{epochs} - "
              f"Train Loss: {avg_train_loss:.6f} - "
              f"Directional Accuracy: {directional_accuracy:.2f}%")

    return directional_accuracy

print(train_and_evaluate_model(hidden_dim=64, num_layers=8, learning_rate=0.00011509786190170734, epochs=20))


########################################
# Step 5: Using the Model with Directional Accuracy
########################################
# Ensure the test dataset is not empty
if len(test_dataset) > 0:
    # Use the model to evaluate predictions across the entire test set for directional accuracy
    import numpy as np

    model.eval()
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch).cpu().numpy()
            all_predictions.append(predictions)
            all_actuals.append(y_batch.numpy())

    # Concatenate all batch outputs
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)

    # Calculate directional accuracy across the test set
    predicted_direction = np.sign(all_predictions)
    actual_direction = np.sign(all_actuals)
    correct_direction = (predicted_direction == actual_direction)
    directional_accuracy = np.mean(correct_direction) * 100

    print(f"Directional Accuracy over Test Set: {directional_accuracy:.2f}%")

    try:
        # Get the last sample from the test dataset for price prediction
        X_last, y_last = test_dataset[len(test_dataset)-1]

        # Check if X_last has the correct shape
        if X_last.shape[0] != seq_length:
            raise ValueError(f"Expected X_last to have sequence length {seq_length}, "
                             f"but got {X_last.shape[0]}")

        # Prepare the input for the model
        X_last = X_last.unsqueeze(0).to(device)  # Shape: (1, seq_length, input_dim)

        # Set the model to evaluation mode
        model.eval()

        # Make the prediction for the last sample
        with torch.no_grad():
            predicted_returns = model(X_last).cpu().numpy()[0]  # Shape: (N,)

        # Get the actual returns for the last sample
        actual_returns = y_last.numpy()  # Shape: (N,)

        # Retrieve the index corresponding to the last sample in the original price data
        last_sample_idx = train_size + len(test_dataset) + seq_length - 1
        if last_sample_idx >= len(prices_df):
            last_sample_idx = len(prices_df) - 1

        # Get the last actual prices
        current_prices = prices_df.iloc[last_sample_idx].values  # Shape: (N,)

        # Calculate predicted prices using predicted returns
        predicted_prices = current_prices * (1 + predicted_returns)

        # Create DataFrames for current and predicted prices
        df_current_prices = pd.DataFrame({
            'ticker': tickers,
            'current_price': current_prices
        })

        df_predicted_prices = pd.DataFrame({
            'ticker': tickers,
            'predicted_price': predicted_prices
        })

    except IndexError:
        print("Error: test_dataset is empty. Cannot make predictions.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
else:
    print("Error: test_dataset has no samples.")


# Simple LP no risk management no nothing
simple_lp = DataProcessor(df_current=df_current_prices, df_predicted=df_predicted_prices)
predicted_pct = predicted_returns * 100
actual_pct = actual_returns * 100
# Merge DataFrames and calculate differences
merged_df = simple_lp.merge_dataframes()

predicted_pct_df = pd.DataFrame(predicted_pct,columns=['Predicted_Percentage'])
actual_pct_df = pd.DataFrame(actual_pct,columns=['Actual_Percentage'])

final_df = pd.concat([merged_df,predicted_pct_df, actual_pct_df], axis=1)
# final_df = pd.merge(predicted_pct_df, actual_pct_df, on='ticker', how='inner', suffixes=('_current', '_predicted'))


class bcolors:
    # Text colors
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'  # Resets all styles
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

    # Bright text colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Bright background colors
    BG_BRIGHT_BLACK = '\033[100m'
    BG_BRIGHT_RED = '\033[101m'
    BG_BRIGHT_GREEN = '\033[102m'
    BG_BRIGHT_YELLOW = '\033[103m'
    BG_BRIGHT_BLUE = '\033[104m'
    BG_BRIGHT_MAGENTA = '\033[105m'
    BG_BRIGHT_CYAN = '\033[106m'
    BG_BRIGHT_WHITE = '\033[107m'
    
print('\n\n\n')
print(bcolors.BRIGHT_GREEN+'-'*40)
print("NN RESULTS: \n")
print(tabulate(final_df, headers='keys', tablefmt='psql'))

sign_df = pd.DataFrame()

sign_df['pred_sign'] = np.where(final_df['Predicted_Percentage'] > 0, 1, np.where(final_df['Predicted_Percentage'] < 0, -1, 0))
sign_df['actual_sign'] = np.where(final_df['Actual_Percentage'] > 0, 1, np.where(final_df['Actual_Percentage'] < 0, -1, 0))

# Check if sign is the same (ignoring zeros if needed)
sign_df['correct_direction'] = (sign_df['pred_sign'] == sign_df['actual_sign'])

# Count how many are correct
correct_count = sign_df['correct_direction'].sum()
print(f"Direction correct count: {correct_count} correct out of  {len(tickers)}" )
total_count   = len(sign_df)

direction_accuracy = correct_count / total_count
print(f"Direction Accuracy: {direction_accuracy*100:.2f}%")

print('-'*40)
print('\n\n\n'+bcolors.ENDC)


    
# Perform optimization with a given budget
budget = 10000  # Example budget
try:
    optimization_result = simple_lp.optimize_portfolio(budget=budget)
    print(bcolors.OKCYAN+'-'*40)
    print("Results of simple LP optimization no risk management no nothing:")
    print("\nOptimization Results:")
    print(f"Status: {optimization_result['status']}")
    print(f"Total Invested: ${optimization_result['total_invested']:.2f}")
    print(f"Total Expected Return: ${optimization_result['total_expected_return']:.2f}")
    print("Investment Allocation:")
    flag = False
    
    for stock, qty in optimization_result['investment'].items():
        print(bcolors.OKCYAN+f"  {stock}: {qty:.2f} shares" )
        print(f"Actual return: {optimization_result['total_invested']*actual_returns[tickers.index(stock)]:.2f}"+ bcolors.ENDC)
        flag = True
    
    if flag==False:
        print(bcolors.FAIL+"ALGORITHM CHOSE NO STOCKS"+bcolors.ENDC)
    
    print(bcolors.OKCYAN + '-'*40 + bcolors.ENDC+'\n\n\n')
        
except Exception as e:
    print(f"Error during optimization: {e}")


# More complex LP with diversification and sector constraints 
budget = 10000                                     # Total budget available
min_stocks = 3                                    # Minimum number of different stocks
sector_cap = 0.4                                  # Maximum 40% budget in any one sector

# Call the optimizer
results = optimize_portfolio(predicted_returns, current_prices, tickers, budget, min_stocks)


expected_returns_sum = 0
actual_returns_sum = 0

# Beautifully print the results
print(bcolors.OKBLUE+"-" * 40)
print("Optimized Portfolio Allocation:")
data = {
    'ticker': [res['ticker'] for res in results],
    'shares': [res['shares'] for res in results],
    'investment': [res['investment'] for res in results],
    'expected_return': [res['investment'] * predicted_returns[tickers.index(res['ticker'])] for res in results],
    'actual_return': [res['investment'] * actual_returns[tickers.index(res['ticker'])] for res in results]
}

print(tabulate(data, headers='keys', tablefmt='psql'))
for res in results:
    expected_return = res['investment'] * predicted_returns[tickers.index(res['ticker'])]
    actual_return = res['investment'] * actual_returns[tickers.index(res['ticker'])]
    
    expected_returns_sum += expected_return
    actual_returns_sum += actual_return
     
print(f"Total Expected Return: ${expected_returns_sum:.5f}")
print(f"Total Actual Return: ${actual_returns_sum:.5f}")
print("-" * 40 +bcolors.ENDC+'\n\n\n')   


current_prices_dict = dict(zip(tickers, current_prices))
predicted_prices_dict = dict(zip(tickers, predicted_prices))
expected_returns_dict = dict(zip(tickers, predicted_returns))
actual_returns_dict = dict(zip(tickers, actual_returns))


spy = yf.download('SPY', start=start_date, end=end_date, progress=True)

spy = spy['Close']
spy_np = spy.to_numpy()
spy_returns = spy_np[1:] / spy_np[:-1] - 1.0

# Ensure arrays have the same number of rows
min_length = min(len(returns), len(spy_returns))
returns = returns[:min_length, :]
spy_returns = spy_returns[:min_length]


betas = np.zeros(returns.shape[1])
var_spy = np.var(spy_returns, ddof=1)  # Using sample variance (ddof=1)


for i in range(returns.shape[1]):
    # Create a 2D array with SPY returns and ticker returns as rows
    combined = np.column_stack((spy_returns, returns[:, i])).T  # shape becomes (2, T)
    cov_matrix = np.cov(combined, ddof=1)
    cov_spy_ticker = cov_matrix[0, 1]
    betas[i] = cov_spy_ticker / var_spy


# make a dictionary with the corresponding beta fora each stock
beta_dict = dict(zip(tickers, betas))

lamda_penalty = 0.1
allocations,status = optimize_portfolio_with_risk(
    tickers = tickers,
    current_prices = current_prices_dict,
    predicted_returns = expected_returns_dict,
    risks = beta_dict,
    budget = budget,
    lambda_penalty = lamda_penalty,
    fully_invest = False,
    sector_constraints = None,
    max_sector_allocation = None,
    solver = None,
    verbose = False
)
    

risk_results = pd.DataFrame(columns=['ticker','shares', 'investment', 'expected_return', 'actual_return'])

actual_returns_sum = 0
expected_returns_sum = 0

# Print the results.
# print(f"Optimization Status: {status}")
for ticker, alloc in allocations.items():
    if alloc > 0:
        shares = alloc
        investment = shares * current_prices_dict[ticker]
        expected_return = investment * expected_returns_dict[ticker]
        actual_return = investment * actual_returns_dict[ticker]
        risk_results_temp = {
            'ticker': ticker,
            'shares': shares,
            'investment': investment,
            'expected_return': expected_return,
            'actual_return': actual_return
        }
        risk_results.loc[len(risk_results)] = risk_results_temp
        actual_returns_sum += actual_return
        expected_returns_sum += expected_return
        
print(bcolors.BRIGHT_CYAN+"-" * 40)
print("Optimized Portfolio Allocation with Risk Management:")
print(tabulate(risk_results, headers='keys', tablefmt='psql'))
print(f"Total Expected Return: ${expected_returns_sum:.5f}")
print(f"Total Actual Return: ${actual_returns_sum:.5f}")
print("-" * 40 +bcolors.ENDC+'\n\n\n')
