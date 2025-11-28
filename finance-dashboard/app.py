import streamlit as st
import yfinance as yf
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# --- TITLE ---
st.title("Rujeh's Financial Dashboard 📈")
st.write("Analyze stock risk factors (Beta) using Python.")

# --- INPUT ---
stock_ticker = st.text_input("Enter Stock Ticker", "TM") # Default is Toyota
market_ticker = "^GSPC" # S&P 500 Index

# --- ENGINE ---
if stock_ticker:
    try:
        # 1. Get Data from Yahoo Finance
        data = yf.download([stock_ticker, market_ticker], start="2023-01-01", end="2024-01-01")['Close']
        
        # 2. Calculate Returns (Daily % change)
        returns = data.pct_change().dropna()
        returns.columns = ['Market', 'Stock']

        # 3. Calculate Beta (Risk)
        X = returns['Market']
        y = returns['Stock']
        X1 = sm.add_constant(X)
        model = sm.OLS(y, X1)
        results = model.fit()
        
        beta = results.params['Market']
        r2 = results.rsquared

        # --- DISPLAY ---
        col1, col2 = st.columns(2)
        col1.metric("Beta (Risk)", f"{beta:.2f}")
        col2.metric("R-Squared (Correlation)", f"{r2:.2f}")
        
        if beta > 1.0:
            st.warning(f"⚠️ {stock_ticker} is MORE volatile than the market.")
        else:
            st.success(f"✅ {stock_ticker} is LESS volatile than the market.")

        # --- CHART ---
        st.subheader("Regression Plot")
        fig, ax = plt.subplots()
        sns.regplot(x='Market', y='Stock', data=returns, ax=ax, line_kws={'color':'red'})
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Detailed Error: {e}")