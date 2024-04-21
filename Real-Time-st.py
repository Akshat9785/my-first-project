import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

st.title('Real-Time Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Function to fetch real-time stock data
def get_real_time_data(stock_symbol):
    try:
        stock_data = yf.Ticker(stock_symbol)
        return stock_data.history(period="1d", interval="1m")
    except:
        return None

df = get_real_time_data(user_input)

if df is not None:
    # Display the first few rows of the DataFrame
    st.subheader('Real-Time Data')
    st.write(df)

    # Visualization
    st.subheader('Real-Time Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'])
    st.pyplot(fig)
else:
    st.subheader('Invalid stock symbol. Please enter a valid one.')

st.write('Note: Real-time data is fetched for the last trading day. The app updates automatically.')
