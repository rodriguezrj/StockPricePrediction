# StockPricePrediction
We Developed an AI-powered Stock Predictor to forecast 1-year, 5-year, and 10-year stock values based on historical closing day data. With this predictor we aim to provide investment guidance to a broad audience, starting with our class and instructors.

 <img align="center" width="900" height="300" src='![alt text](UMPI-Stock-Market-Projection-Software.png)'>

# Table of Contents
- [Introduction](#introduction)

- [Installation](#introduction)

- [Data Download](#data-download)

- [50-day Rolling Average](#50-day-rolling-average)

- [Data Processing](#data-processing)

- [Model Training](#model-training)

    - [LSTM Model](#lstm-model)

    - [GRU Model](#gru-model)

    - [Prophet Model](#prophet-model)

- [Model Evaluation](#model-evaluation)

- [Plotting Results](#plotting-results)

- [Sentiment Analysis](#sentiment-analysis)

- [Performance Comparison](#performance-comparison)

- [Conclusion](#conclusion)

- [Resources](#resources)

# Introduction

This project aims to predict stock prices using machine learning models, specifically LSTM (Long Short-Term Memory), GRU (Gated Recurrent Unit), and Prophet. The project involves downloading historical stock data, preprocessing it, and training models to forecast future stock prices.
# Installation
To run this project, you need to install the following Python packages:
yfinance: For downloading historical stock data.
numpy and pandas: For data manipulation.
matplotlib: For plotting results.
scikit-learn: For preprocessing and evaluation metrics.
tensorflow: For building and training LSTM and GRU models.
prophet: For building and training the Prophet model.
vaderSentiment: For performing sentiment analysis on news headlines.
# Data Download
We used the yfinance library to download historical stock data for Apple (AAPL), Google (GOOGL), and Microsoft (MSFT). This data includes the daily closing prices from January 2010 to January 2024. The closing price is the most commonly used price for financial analysis as it reflects the final price at which the stock was traded on a particular day.
# 50-day Rolling Average
A 50-day moving average is calculated for each stock. This is a widely used indicator in technical analysis that smooths out price data by creating a constantly updated average price. It helps in identifying trends over a medium-term period. We handle missing values by backfilling, which means filling the missing values with the next available value in the dataset.
# Data Processing
This section involves several steps:
Processing Stock Data: The raw stock data is smoothed using a 50-day moving average. After smoothing, missing values are handled by backfilling. The data is then scaled to a range between 0 and 1 using MinMaxScaler. Scaling is important in machine learning as it standardizes the range of independent variables or features of data.
Creating Datasets for LSTM: The data is formatted into sequences suitable for training LSTM models. A time step parameter determines the number of previous days used to predict the next day’s stock price. This transformation is crucial for time series forecasting as it helps the model learn temporal dependencies.
Splitting Data into Training and Testing Sets: The data is divided into training and testing sets. The training set is used to train the model, while the testing set is used to evaluate its performance. A typical split ratio is 80-20, meaning 80% of the data is used for training and 20% for testing.
# Model Training
# LSTM Model
LSTM (Long Short-Term Memory) networks are a type of recurrent neural network (RNN) capable of learning long-term dependencies. They are particularly effective for time series prediction due to their ability to remember information for long periods. The model is trained to minimize the mean squared error between predicted and actual stock prices.
# GRU Model
GRU (Gated Recurrent Unit) networks are a variation of LSTM that combine the forget and input gates into a single update gate. GRUs have fewer parameters than LSTMs, which can make them faster to train and perform well on smaller datasets. Like LSTMs, they are trained to minimize the mean squared error.
# Prophet Model
Prophet is an open-source forecasting tool developed by Facebook. It is designed for time series data that have strong seasonal effects and several seasons of historical data. Prophet is intuitive to use and works well with missing data and outliers. It automatically detects yearly, weekly, and daily seasonality, making it suitable for business forecasting.
# Model Evaluation 
The performance of each model is evaluated using the following metrics:
RMSE (Root Mean Squared Error): Measures the square root of the average squared differences between predicted and actual values. Lower RMSE indicates better model performance.
MAE (Mean Absolute Error): Measures the average absolute differences between predicted and actual values. Like RMSE, lower MAE indicates better performance, but it is less sensitive to outliers.
R-squared (R²): Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. Higher R² values indicate better model performance, with 1 indicating a perfect fit.
# Plotting Results
The predicted stock prices are plotted against the actual stock prices to visualize the performance of the models. These plots help in understanding how well the models are able to follow the actual stock price trends.
X-axis: Represents the date.
Y-axis: Represents the stock price.
Blue Line: Actual stock prices.
Orange Line: Predicted stock prices by the model.
# Sentiment Analysis
Sentiment analysis is performed on news headlines related to the stocks using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool. VADER is particularly effective for analyzing social media text. It provides a sentiment score ranging from -1 (very negative) to +1 (very positive).
Collect News Data: News headlines related to the stocks are collected.
Calculate Sentiment Scores: VADER is used to calculate sentiment scores for each headline.
Integrate Sentiment with Stock Prices: The sentiment scores are integrated with stock price data to enhance predictions.
# Performance Comparison
The performance of the LSTM, GRU, and Prophet models is compared using bar plots of RMSE, MAE, and R-squared metrics. This helps in determining which model provides the best predictions for our stock data.
RMSE Comparison: Lower RMSE values indicate better model performance.
MAE Comparison: Lower MAE values indicate better model performance.
R-squared Comparison: Higher R² values indicate better model performance.
# Conclusion
This project demonstrates how to use LSTM, GRU, and Prophet models to predict stock prices and analyze the impact of sentiment on stock performance. By comparing the performance of these models using RMSE, MAE, and R-squared metrics, we can determine which model provides the best predictions for our stock data.

The LSTM model performed the best with the lowest RMSE and MAE and the highest R², indicating it provides the most accurate predictions and explains the most variance in the stock prices followed by the GRU model and lastly the prophet model.
# Resources
YFinance 