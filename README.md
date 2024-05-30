# Market-Prediction-Using-LSTM

# G-Research Crypto Forecasting Competition

## Overview

This project involves building machine learning models to forecast returns of 14 popular cryptocurrencies on a minute-by-minute basis. The work is part of the [G-Research Crypto Forecasting competition](https://www.kaggle.com/competitions/g-research-crypto-forecasting) hosted on Kaggle. This competition challenges participants to predict future cryptocurrency returns using historical trading data.

## Data Description

The dataset provided by G-Research contains minute-by-minute trading data of 14 cryptocurrencies. It includes the following features:

- `timestamp`: Unix timestamp in seconds.
- `Asset_ID`: An identifier for each crypto asset.
- `Count`: Number of trades in the interval.
- `Open`: Opening price.
- `High`: Highest price.
- `Low`: Lowest price.
- `Close`: Closing price.
- `Volume`: Trading volume.
- `VWAP`: Volume Weighted Average Price.
- `Target`: Residual log-returns for the asset over a 15 minute horizon.

## Project Structure

### Data Preprocessing

- Handling missing data by identifying gaps in timestamps.
- Re-indexing datasets to ensure continuous time intervals.
- Data type optimization to reduce memory usage.

### Exploratory Data Analysis

- Visualizing trading dynamics using candlestick charts.
- Correlation analysis among different cryptocurrencies.

### Feature Engineering

- Generating features like log returns and various technical indicators such as moving averages and volume weighted indicators.
- Cross-asset features to capture market-wide effects.

### Model Building

- Utilizing LightGBM for baseline models with detailed custom metric (Pearson correlation coefficient) for evaluation.
- Advanced model exploration with LSTM and TCN (Temporal Convolutional Networks) models to capture time-series dependencies.

### Evaluation

- Splitting data into training and validation sets to measure model performance.
- Employing time series cross-validation techniques.

## Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `lightgbm`
- `keras`
- `tcn` - Temporal Convolutional Network


