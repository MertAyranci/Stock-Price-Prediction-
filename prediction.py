import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt

# 1) Download AAPL adjusted for splits & dividends
ticker     = "AAPL"
start_date = "2000-01-01"
df = yf.download(
    ticker,
    start=start_date,
    auto_adjust=True,
    progress=False
)

# 2) Ensure a business‐day index & fill any holes
df = df.asfreq('B')
df = df.fillna(method='ffill')

# 3) Feature engineering
df['Return'] = df['Close'].pct_change()
df['MA10']    = df['Close'].rolling(10).mean()
df.dropna(inplace=True)

# 4) Train/test split   
split    = int(len(df) * 0.8)
train_df = df.iloc[:split]
test_df  = df.iloc[split:]

# 5) Scaling
scaler       = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[['Close', 'Return', 'MA10']])
test_scaled  = scaler.transform(test_df[['Close', 'Return', 'MA10']])

# (Optional) Quick sanity‑check printouts
print("Training shape:", train_scaled.shape)
print("Testing shape: ", test_scaled.shape)
print(train_df.head(), "\n", train_scaled[:5])

# 6) From here you’d build your PyTorch Dataset/DataLoader, model, etc.



train_scaled_df = pd.DataFrame(
    train_scaled,
    index=train_df.index,
    columns=['Close','Return','MA10']
)
test_scaled_df = pd.DataFrame(
    test_scaled,
    index=test_df.index,
    columns=['Close','Return','MA10']
)

# Plot scaled Close price
plt.figure(figsize=(12,6))
plt.plot(train_df.index, train_df['Close'], label='Train (raw Close)')
plt.plot(test_df.index,  test_df['Close'],  label='Test (raw Close)')
plt.title('Apple Close Price: Train vs. Test')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("raw_close_train_test.png")
plt.close()
print("Saved raw price plot to raw_close_train_test.png")