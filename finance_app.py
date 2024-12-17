import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Title and Description
st.title("Stock Price Prediction App")
st.markdown("Predict upcoming stock prices using machine learning (LSTM).")

# Input Section
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

if st.sidebar.button("Fetch Data"):
    # Fetch historical data
    st.write("Fetching data...")
    data = yf.download(ticker, start=start_date, end=end_date)
    st.write("Historical Stock Prices", data)
    st.line_chart(data['Close'])

# Data Preprocessing
@st.cache
def preprocess_data(data):
    data = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# Build LSTM Model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Prepare Data for Training
if st.sidebar.button("Train Model"):
    data_scaled, scaler = preprocess_data(data)
    seq_len = 60
    X_train, y_train = [], []
    for i in range(seq_len, len(data_scaled)):
        X_train.append(data_scaled[i-seq_len:i])
        y_train.append(data_scaled[i])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Train Model
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, batch_size=32, epochs=5)

    st.write("Model Trained Successfully!")

# Predictions
if st.sidebar.button("Predict Next Week"):
    last_60_days = data_scaled[-60:]
    last_60_days = np.expand_dims(last_60_days, axis=0)
    predictions = model.predict(last_60_days)
    predictions = scaler.inverse_transform(predictions)
    st.write("Predicted Prices for the Next Week", predictions)

    # CSV Download
    pred_df = pd.DataFrame(predictions, columns=['Predicted Price'])
    st.download_button(label="Download Predictions as CSV", data=pred_df.to_csv(index=False), file_name="predictions.csv")
