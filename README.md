# Microsoft Stock Price Prediction (LSTM)

This project predicts Microsoft stock closing prices using an LSTM (Long Short-Term Memory) neural network.

## 📊 Features
- Loads historical Microsoft stock price data from CSV
- Preprocesses data for time series forecasting
- Trains an LSTM model using TensorFlow/Keras
- Plots:
  - Actual vs Predicted closing prices
  - (Optional) Correlation heatmap

## 📂 Project Structure
MyLSTMProject/
│-- main.py # Main script for data processing, model training, and plotting
│-- MicrosoftStock.csv # Historical stock data
│-- requirements.txt # Python dependencies
│-- README.md # Project documentation
│-- .gitignore # Git ignore rules


## 🛠 Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

python main.py
