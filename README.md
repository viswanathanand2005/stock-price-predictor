# 📈 Microsoft Stock Price Prediction using LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to forecast the closing price of Microsoft (MSFT) stock based on historical data.

## 🌟 Features

* Loads historical stock price data for Microsoft from a CSV file.
* Preprocesses and scales the data for time series forecasting.
* Builds and trains an LSTM model using TensorFlow and Keras.
* Visualizes the results by plotting:
    * Actual vs. Predicted closing prices.
    * (Optional) A correlation heatmap of the dataset features.

***

## 📂 Project Structure

Here is the file organization for this project:
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
  
