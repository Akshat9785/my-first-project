import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go

from datetime import date


start = '2015-01-01'
end = '2025-12-31'

# Set page title and favicon
st.set_page_config(
    page_title="Stock Trend Prediction",
    page_icon=":chart_with_upwards_trend:",
)

# Add custom CSS for better styling
st.write('<style>body { background-color: #f0f0f0; }</style>', unsafe_allow_html=True)

# Display a large title
st.title("Stock Trend Prediction")


# Stock Ticker Input
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Date Range Selector
start_date = st.date_input('Select start date', date(2015, 1, 1))
end_date = st.date_input('Select end date', date(2025, 12, 31))

# Fetch data based on user's selection
if user_input:
    df = yf.download(user_input, start=start_date, end=end_date)

    # Update the title based on the selected date range
    title = f'Data from {start_date.year} - {end_date.year}'
    st.subheader(title)



# Display the first few rows of the DataFrame
print(df.head())

# Describing Data
#st.subheader('Data from 2013 - 2022')
st.write(df.describe())



# Visulaization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, color='green')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, color='red')
plt.plot(df.Close, color='green')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, color='red')
plt.plot(ma200)
plt.plot(df.Close, color='green')
st.pyplot(fig)


# Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


# Load My Model

model = load_model('keras_model.h5')

# Testing Part 

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array (y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Final Graph

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


# Feedback Section
st.subheader('User Feedback')
user_feedback = st.text_area('Please share your feedback here:')
if st.button('Submit Feedback'):
    # You can add code here to save or process the user's feedback
    st.success('Thank you for your feedback!')
