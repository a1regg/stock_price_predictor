# LSTM Stock/Asset Price Predictor

The project that demonstrates how to use **Long Short-Term Memory (LSTM)** networks for stock (or other financial asset) price prediction. It provides two interfaces:
1. A **Streamlit**-based web application (`model.py`) to visualize training/test predictions and future forecasts.
2. A **PyQt6** desktop application (`gui.py`) with an interactive user interface for generating predictions.


---

## Features
- **Historical Data Fetching**: Pulls price data from Yahoo! Finance.
- **LSTM Model**: Utilizes a multi-layer LSTM with dropout for sequence prediction.
- **Configurable Parameters**: Window size (sequence length), future prediction days, ticker symbol, and date ranges.
- **Interactive Plots**: 
  - In Streamlit, plots are shown directly on the web app.
  - In PyQt6, two plots appear side by side (test-set results and future forecast).
- **Error Handling**: Alerts users when no data is found.

---

## Project Structure
```
project_folder/
├── src/
│   ├── model.py                # Core model training + Streamlit UI 
│   ├── gui.py                  # PyQt6 GUI for real-time stock prediction
├── requirements.txt            # Python dependencies for the project
└── README.md                   # Project documentation
```

---

## Installation

1. **Clone the Repository**
```
  git clone https://github.com/a1regg/stock_price_predictor.git
  cd stock_price_predictor
```

2. **Create virtual environment(optional):**
   ```
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    # or
    venv\Scripts\activate.bat # For Windows
   ```
3. Install dependecies:
   `pip install -r requirements.txt`

## Usage

### Using Streamlit App

1. Run
  ```
  cd src
  streamlit run model.py
  ```
3. Open your browser at the URL shown in the terminal (usually `http://localhost:8501`).
4. Enter a valid ticker symbol (e.g., `AAPL`), set your desired date range and parameters, then click Predict.

### Using PyQt6 GUI

1. Run
   ```
   cd src
   python gui.py
   ```
3. A desktop window will appear with text fields and spin boxes for your inputs.
4. Click **Predict** to train the model and view the plots.

   
## Acknowledgments

This project makes use of the **[yfinance](https://pypi.org/project/yfinance/)** library, which greatly simplifies accessing historical market data from Yahoo Finance.
