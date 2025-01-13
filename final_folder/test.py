
import yfinance as yf

data = yf.download("AAPL", start="2000-01-01", end="2021-01-01", auto_adjust=True)

new_data = data["Close"] 
print(data)  # This will show the adjusted close prices