import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# For LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ----------------------------------------------------
# 1) Helper function to create sequences
# ----------------------------------------------------
def create_sequences(data, window_size=60):
    """
    data: NumPy array of shape (num_samples, 1)
    window_size: Number of time steps per input sequence
    Returns:
      X, y - where X has shape (num_sequences, window_size, 1)
              and y has shape (num_sequences,)
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# ----------------------------------------------------
# 2) Model training & prediction function
# ----------------------------------------------------
def train_and_predict(ticker, start_date, end_date, window_size=60, future_days=7):
    """
    1. Downloads historical data
    2. Trains LSTM model (80% train / 20% test)
    3. Plots test-set predictions
    4. Forecasts 'future_days' beyond the last known date
    5. Returns two figures:
        fig_test  -> Historical test performance
        fig_future -> Future forecast
    """
    # Download data from Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Basic error handling if no data is returned
    if df.empty:
        st.error("No data found for the given ticker/date range. Please try again.")
        return None, None
    
    # Focus on 'Close' price
    df = df[['Close']].dropna()
    
    # Scale data (0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Create sequences for training
    X, y = create_sequences(scaled_data, window_size=window_size)
    # X shape => (samples, window_size)
    # y shape => (samples,)

    # Reshape X for LSTM: (samples, timesteps, features=1)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Train-test split (80% / 20%)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build a simple LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(
        X_train, y_train, 
        epochs=10,           
        batch_size=32, 
        validation_split=0.1,
        verbose=1,
        shuffle=False
    )
    
    # ------------------------
    # A) Predict on Test Set
    # ------------------------
    test_predictions_scaled = model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions_scaled)
    test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Plot results on Test Set
    fig_test, ax_test = plt.subplots(figsize=(10, 4))
    ax_test.plot(test_actual, color='black', label='Real Price (Test)')
    ax_test.plot(test_predictions, color='green', label='Predicted Price (Test)')
    ax_test.set_title(f"{ticker}: Test-Set Prediction")
    ax_test.set_xlabel("Test Samples")
    ax_test.set_ylabel("Price")
    ax_test.legend()

    # ------------------------
    # B) Predict 'future_days'
    # ------------------------
    # We'll use the last 'window_size' points from the entire dataset
    # to predict forward day by day.
    last_sequence = scaled_data[-window_size:]  # shape = (window_size, 1)

    future_preds_scaled = []
    for _ in range(future_days):
        # Reshape last_sequence to (1, window_size, 1)
        pred_input = last_sequence.reshape((1, window_size, 1))
        pred_out = model.predict(pred_input)  # shape = (1, 1)
        future_preds_scaled.append(pred_out[0, 0])
        
        # Shift the window by 1, append the prediction
        # new shape still (window_size, 1)
        last_sequence = np.append(last_sequence[1:], [[pred_out[0,0]]], axis=0)
        
    # Inverse transform future predictions
    future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1,1))
    
    # Build a date range for the forecast (1 day after the last date in df)
    last_known_date = df.index[-1]
    forecast_index = pd.date_range(start=last_known_date + pd.Timedelta(days=1),
                                   periods=future_days,
                                   freq='B')  # 'B' = business days
    # Create a small DataFrame for plotting
    df_future = pd.DataFrame(data=future_preds, index=forecast_index, columns=['Forecast'])

    # Plot historical + forecast
    fig_future, ax_future = plt.subplots(figsize=(10, 4))
    ax_future.plot(df.index, df['Close'], label='Historical', color='black')
    ax_future.plot(df_future.index, df_future['Forecast'], label='Future Forecast', color='blue')
    ax_future.set_title(f"{ticker}: {future_days}-Day Future Forecast")
    ax_future.set_xlabel("Date")
    ax_future.set_ylabel("Price")
    ax_future.legend()

    return fig_test, fig_future


# ----------------------------------------------------
# 3) Streamlit UI
# ----------------------------------------------------
def main():
    st.title("Generic Stock (or Asset) Price Predictor")

    # Ticker input (e.g., 'AAPL', 'TSLA', 'GC=F' for gold)
    ticker = st.text_input("Enter Ticker Symbol:", value="AAPL")

    # Date inputs
    start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
    end_date   = st.date_input("End Date",   value=pd.to_datetime("2025-01-01"))

    # Window size input
    window_size = st.number_input("Window Size (days for LSTM sequence)",
                                  min_value=1, max_value=365, value=60)
    
    # How many future days to forecast
    future_days = st.number_input("Number of Future Days to Predict",
                                  min_value=1, max_value=60, value=7)
    
    # Predict button
    if st.button("Predict"):
        fig_test, fig_future = train_and_predict(
            ticker,
            start_date,
            end_date,
            window_size,
            future_days
        )
        # Display the test-set prediction figure
        if fig_test:
            st.pyplot(fig_test)
        # Display the future forecast figure
        if fig_future:
            st.pyplot(fig_future)

if __name__ == "__main__":
    main()
