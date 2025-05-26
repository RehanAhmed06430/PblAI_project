import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense
import os
import datetime

st.set_page_config(page_title="📈 Stock Forecast App", layout="wide")

# ========================================
# Utility Functions
# ========================================

@st.cache_data
def load_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def preprocess_data(df, feature='Close', sequence_len=60):
    values = df[[feature]].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(sequence_len, len(scaled)):
        X.append(scaled[i-sequence_len:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_model(model_type, input_shape):
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50))
    elif model_type == "GRU":
        model.add(GRU(50, return_sequences=True, input_shape=input_shape))
        model.add(GRU(50))
    else:
        model.add(Dense(64, activation='relu', input_shape=input_shape))
        model.add(Dense(32, activation='relu'))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_days(model, last_sequence, steps, scaler):
    future_preds = []
    seq = last_sequence.reshape(1, -1, 1)

    for _ in range(steps):
        pred = model.predict(seq, verbose=0)[0]
        future_preds.append(pred)
        seq = np.append(seq[:, 1:, :], [[pred]], axis=1)

    return scaler.inverse_transform(np.array(future_preds))

def plot_chart(title, actual, predicted, future=None, future_dates=None):
    plt.figure(figsize=(10, 4))
    plt.plot(actual, label="Actual", color='blue')
    plt.plot(predicted, label="Predicted", color='red')
    if future is not None and future_dates is not None:
        plt.plot(future_dates, future, label="Forecast", color='orange')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

# ========================================
# App UI
# ========================================

st.title("📊 Stock Price Prediction & Forecasting")

# Sidebar inputs
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2023, 12, 31))
feature = st.sidebar.selectbox("Feature to Predict", ["Close", "Open", "High", "Low", "Volume"])
model_type = st.sidebar.selectbox("Model Type", ["LSTM", "GRU", "Dense"])
future_days = st.sidebar.slider("Forecast Future Days 🗓️", 1, 60, 30)
train_button = st.sidebar.button("▶️ Run Prediction")

# Models folder
os.makedirs("models", exist_ok=True)
model_filename = f"models/{ticker}_{model_type}_{feature}.h5"

if train_button:
    st.subheader(f"Loading data for {ticker}...")
    
    df = load_stock_data(ticker, start_date, end_date)
    
    if df.empty:
        st.error("❌ Failed to load stock data.")
    else:
        st.success("✅ Data Loaded!")
        st.line_chart(df[feature], use_container_width=True)

        st.subheader("📊 Preprocessing + Training...")
        X, y, scaler = preprocess_data(df, feature)
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        # Try loading saved model; train if not found
        if os.path.exists(model_filename):
            model = load_model(model_filename)
            st.info("🤖 Loaded trained model from disk.")
        else:
            model = build_model(model_type, (X_train.shape[1], 1))
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
            model.save(model_filename)
            st.success("✅ Model trained and saved.")

        # Predict
        y_pred = model.predict(X_test, verbose=0)
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        st.subheader("📈 Actual vs Predicted Prices")
        plot_chart(f"{ticker} - {feature} Prediction", y_test_inv, y_pred_inv)

        # Forecast
        st.subheader(f"🔮 Forecasting next {future_days} days...")
        last_seq = X[-1]
        future_preds = forecast_days(model, last_seq, future_days, scaler)
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')

        # Plot forecast
        plot_chart(f"{ticker} Future Forecast", y_test_inv, y_pred_inv, future_preds, future_dates)

        # Show forecast table
        st.subheader("📅 Forecast Table")
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecasted Price": future_preds.flatten()
        }).set_index("Date")
        st.dataframe(forecast_df)