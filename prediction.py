import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# 1) Download AAPL with splits & dividends auto‑adjusted
ticker     = "AAPL"
start_date = "2000-01-01"    # adjust as you like
df = yf.download(
    ticker,
    start=start_date,
    auto_adjust=True,         # <-- this rolls splits & dividends into the prices
    progress=False
)

# 2) Plot the adjusted Close
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="Apple Adjusted Close", color="blue")
plt.title("Apple Adjusted Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
