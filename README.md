# Yes Bank Stock Price Prediction using Machine Learning

This project presents a comprehensive end-to-end machine learning solution for predicting the stock price of **Yes Bank** based on its historical performance. The main goal is to help understand stock price patterns, apply predictive modeling, and gain valuable business insights.

## Project Type
EDA | Regression | Supervised Machine Learning

## Problem Statement
The goal is to predict the closing stock price of Yes Bank using past stock market data (Open, High, Low, Close prices). Stock market fluctuations can be unpredictable, but with machine learning and statistical techniques, we aim to model patterns and trends to support better financial decision-making.

## Dataset Overview
-  Contains monthly stock prices of Yes Bank from 2005 to 2020.
-  Features include: Date, Open, High, Low, Close.
-  Cleaned, processed, and visualized for quality modeling.

##  Exploratory Data Analysis (EDA)
We visualized trends and relationships between features:
- Line Plot of Closing Price over Time
- High vs Low Price Line Graph
- Scatter Plots (Open vs Close)
- Year-wise Boxplots
- Correlation Heatmap
- Histogram of Closing Prices

##  Statistical Testing
Conducted multiple hypothesis tests to check data relationships:
1. **ANOVA**: Strong variance across years (P ≈ 7.3e-74)
2. **Kruskal-Wallis**: Reinforced groupwise differences (P ≈ 6.1e-28)
3. **Spearman Correlation**: High correlation with time (ρ ≈ 0.97)

##  Feature Engineering
- Created `Year` column from dates.
- Scaled features using `MinMaxScaler`.
- Checked and handled outliers and missing values.
- Selected relevant features using `SelectKBest`.

##  Models Used & Performance
| Model                 | MSE       | R² Score | After Tuning R² |
|----------------------|-----------|----------|------------------|
| Linear Regression     | 137.23    | 0.9835   | ✅ **0.9835** (Best) |
| Random Forest Regressor | 171.88 | 0.9793   | 0.9809 |
| XGBoost Regressor     | 175.77    | 0.9789   | 0.9777 |

 **Final Model Chosen**: Linear Regression (simpler, robust, highest R²)

##  Evaluation Metrics
- **Mean Squared Error (MSE)**: Measures average squared errors.
- **R² Score**: Explained variance — close to 1.0 means high predictive power.
- Plotted actual vs. predicted values to ensure model sanity.

##  Deployment Ready
- Saved the trained model using `pickle` for deployment.
- Tested model with unseen data to validate predictions.

##  Files Included
- `YesBank_StockPricePrediction.ipynb` — Main project notebook
- `best_model.pkl` — Saved trained ML model
- `README.md` — Project overview
- `requirements.txt` — List of packages used

##  Tools & Libraries
- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost
- Jupyter/Colab Notebook

##  Business Impact
By predicting stock prices with high accuracy, this model can help:
- Financial analysts in risk assessment
- Retail investors in making buy/sell decisions
- Firms in forecasting long-term trends



##  Connect
Made by [Rounak Saha](https://github.com/Rounak36)  


