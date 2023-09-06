import yfinance as yf
import pandas as pd
import ta
import streamlit as st
import mplfinance as mpf 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta

# Streamlit app title
st.title("Candlestick Chart and Signals")

# Dropdown menu for stock selection
selected_stock = st.selectbox("Select Stock Symbol", ["TCS.BO", "RELIANCE.BO", "INFY.BO", "WIPRO.BO", "HDFCBANK.BO"])

# Date selection for the end date (only end date will be selected)
end_date = st.date_input("Select End Date")

# Calculate the start date by subtracting one year from the end date
start_date = end_date - timedelta(days=365)

# Check if the user has selected a valid end date
if start_date < end_date:
    try:
        # Fetch historical stock data for the past year
        stock = yf.Ticker(selected_stock)
        historical_data = stock.history(start=start_date, end=end_date)

        # Calculate RSI
        rsi_length = 14
        historical_data['RSI'] = ta.momentum.RSIIndicator(historical_data['Close'], window=rsi_length).rsi()

        # Calculate MACD
        macd_fast_length = 12
        macd_slow_length = 26
        historical_data['MACD'] = ta.trend.MACD(historical_data['Close'], window_slow=macd_slow_length, window_fast=macd_fast_length).macd()

        # Calculate MACD EMA
        macd_ema_length = 6
        historical_data['MACD_EMA'] = historical_data['MACD'].ewm(span=macd_ema_length, adjust=False).mean()

        # Calculate SMA of RSI values with length 14
        historical_data['SMA_RSI'] = historical_data['RSI'].rolling(window=14).mean()

        # Create columns to store RSI and MACD signals
        historical_data['RSI_Signal'] = ''
        historical_data['MACD_Signal'] = ''

        # Generate RSI signals
        for i in range(14, len(historical_data)):
            if historical_data['SMA_RSI'][i] < historical_data['RSI'][i]:
                historical_data['RSI_Signal'][i] = 'BULLISH'
            elif historical_data['SMA_RSI'][i] > historical_data['RSI'][i]:
                historical_data['RSI_Signal'][i] = 'BEARISH'

        # Generate MACD signals
        for i in range(26, len(historical_data)):
            if historical_data['MACD'][i] > historical_data['MACD_EMA'][i]:
                historical_data['MACD_Signal'][i] = 'BULLISH'
            elif historical_data['MACD'][i] < historical_data['MACD_EMA'][i]:
                historical_data['MACD_Signal'][i] = 'BEARISH'

        # Display the Candlestick Chart
        if not historical_data.empty:
            st.subheader(f"Candlestick Chart for {selected_stock}")

            # Create a custom plot using mplfinance for the candlestick chart
            custom_style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'lines.linewidth': 1})
            fig, axes = mpf.plot(historical_data, type='candle',
                                 title=f'{selected_stock} Candlestick Chart',
                                 style=custom_style, volume=True, returnfig=True)

            st.pyplot(fig)

            # Create a DataFrame with date and signals
            signal_data = historical_data[['RSI_Signal', 'MACD_Signal']]
            signal_data['Date'] = historical_data.index

            # Display the DataFrame with signals and dates
            # st.subheader("Signals with Dates")
            # st.table(signal_data[['Date', 'RSI_Signal', 'MACD_Signal']])

            # Ensure that there are enough data points for features
            if len(historical_data) >= 40:  # You may adjust this threshold as needed
                # Create a new column for the target labels
                historical_data['Signal'] = ''

                # Encode target labels as numerical values
                label_encoder = LabelEncoder()
                historical_data['Signal'] = label_encoder.fit_transform(historical_data['RSI_Signal'])

                # Select features (RSI and MACD signals)
                features = historical_data[['RSI', 'MACD']]

                # Impute missing values in the feature data
                imputer = SimpleImputer(strategy='mean')  # You can choose another strategy if needed
                features_imputed = imputer.fit_transform(features)

                # Split the data into training and testing sets
                X = features_imputed
                y = historical_data['Signal']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the SVM model
                svm_model = SVC(kernel='linear', C=1)
                svm_model.fit(X_train, y_train)

                # Make predictions
                predictions = svm_model.predict(X_test)

                # Decode numerical predictions back to labels
                predicted_labels = label_encoder.inverse_transform(predictions)

                # Display the predicted signals
                st.subheader("Predicted Signals with Dates (BULLISH or BEARISH)")
                prediction_dates = historical_data.index[-len(y_test):]  # Get corresponding dates for predictions
                prediction_df = pd.DataFrame({'Date': prediction_dates, 'Actual Signal': label_encoder.inverse_transform(y_test), 'Predicted Signal': predicted_labels})
                st.table(prediction_df . tail(30))

                # Calculate accuracy and display classification report
                accuracy = accuracy_score(y_test, predictions)
                classification_rep = classification_report(y_test, predictions, target_names=label_encoder.classes_,
                                                           output_dict=True)

                st.text(f"Accuracy: {accuracy:.2f}")

                # Display the classification report in a table
                st.subheader("Classification Report")
                classification_report_df = pd.DataFrame(classification_rep).transpose()
                st.table(classification_report_df)

            else:
                st.warning("Insufficient data points for feature extraction.")
        else:
            st.error(f"No data available for {selected_stock} in the selected date range.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add a footer
st.text("Data source: Yahoo Finance")