import streamlit as st
import yfinance as yf
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# --- TITLE ---
st.title("Rujeh's Pro Financial Dashboard 🚀")
st.write("Analyze stock risk (Beta) against specific Country Benchmarks.")

# --- INPUTS ---
# We use columns to put the boxes side-by-side
col1, col2 = st.columns(2)

with col1:
    stock_ticker = st.text_input("Enter Stock Ticker", "RMDA.CA")

with col2:
    # The Dropdown Menu
    market_name = st.selectbox(
        "Select The Market",
        ["US Market (S&P 500)", "Egypt Market (EGX 30)"]
    )

# --- LOGIC: MAP NAME TO TICKER ---
if market_name == "US Market (S&P 500)":
    market_ticker = "^GSPC"
else:
    market_ticker = "^CASE30"  # The ticker for the Cairo Index

# --- ENGINE ---
if stock_ticker:
    try:
        # 1. Get Data
        # We start from 2022 to get more data points for Egyptian stocks
        data = yf.download([stock_ticker, market_ticker], start="2022-01-01", end="2024-01-01")['Close']
        
        # 2. Clean Data (Drop days where market was closed)
        returns = data.pct_change().dropna()
        
        # Rename columns to match our math (The order matters!)
        # Yahoo downloads alphabetically. We need to find which column is which.
        # This little trick ensures we grab the right column regardless of name
        stock_col = stock_ticker
        market_col = market_ticker
        
        # Rename for easier code below
        returns = returns.rename(columns={stock_col: 'Stock', market_col: 'Market'})

        # 3. Calculate Beta (Risk)
        X = returns['Market']
        y = returns['Stock']
        X1 = sm.add_constant(X)
        model = sm.OLS(y, X1)
        results = model.fit()
        
        beta = results.params['Market']
        r2 = results.rsquared

        # --- DISPLAY ---
        st.write(f"### Analysis: {stock_ticker} vs {market_name}")
        
        col1, col2 = st.columns(2)
        col1.metric("Beta (Risk)", f"{beta:.2f}")
        col2.metric("Correlation (R²)", f"{r2:.2f}")
        
        # Interpretation Logic
        if beta < 1.0:
            st.success(f"✅ {stock_ticker} is LESS volatile than {market_name}")
        else:
            st.error(f"⚠️ {stock_ticker} is MORE volatile than {market_name}")

        # --- CHART ---
        fig, ax = plt.subplots()
        sns.regplot(x='Market', y='Stock', data=returns, ax=ax, line_kws={'color':'red'})
        st.pyplot(fig)

    except Exception as e:
        st.warning("Could not fetch data. Note: Some Egyptian tickers (EGX 30) have gaps in Yahoo Finance data.")
        st.error(f"Technical Error: {e}")