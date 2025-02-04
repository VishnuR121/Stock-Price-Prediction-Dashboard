from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)

MODEL_PATH = 'models/stock_model.pkl'

# Function to fetch stock data and train the model
def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1y")
        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        data["MA_5"] = data["Close"].rolling(window=5).mean()
        data["Next_Close"] = data["Close"].shift(-1)
        return data.dropna()
    except Exception as e:
        print(f"Error fetching data for symbol {symbol}: {e}")
        return None  # Return None if there's an error

# Function to train and save the model
def train_and_save_model():
    data = get_stock_data('AAPL')  # Use any stock symbol to train the model
    X = data[["Close", "MA_5"]]
    y = data["Next_Close"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, MODEL_PATH)
    print("Model saved!")

# Function to load the pre-trained model
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded!")
        return model
    except FileNotFoundError:
        print("Model not found, training a new model.")
        train_and_save_model()
        return joblib.load(MODEL_PATH)

# Load the model when the app starts
model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    data = get_stock_data(symbol)
    
    if data is None:
        # If no valid data is fetched, show an error message
        error_message = f"Invalid stock symbol or no data found for \"{symbol}\". Please try again."
        return render_template('prediction.html', error_message=error_message)

    # Prepare features and target
    X = data[["Close", "MA_5"]]
    y = data["Next_Close"]
    
    # Use the pre-trained model to make the prediction
    prediction = model.predict(X.tail(1))[0]

    formatted_prediction = f"${prediction:.2f}"
    
    return render_template('prediction.html', symbol=symbol, prediction=formatted_prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)