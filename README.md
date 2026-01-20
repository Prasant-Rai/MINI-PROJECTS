# MINI-PROJECTS - Data Science Portfolio

Complete Machine Learning Projects

## PROJECTS COMPLETED

| Project | Techniques Used | Dataset | Industry |
|---------|-----------------|---------|----------|
| Bitcoin Price Prediction | LSTM + TextBlob Sentiment Analysis | Bitcoin Price + Tweets | Global Finance |
| Breast Cancer Diagnosis | Random Forest Classifier | Wisconsin Breast Cancer | Healthcare |
| Nepal Earthquake Damage | Random Forest Multi-class | Earthquake Damage Data | Disaster Response |
| Global EV Sales Forecast | Linear Regression → Polynomial → ARIMA | IEA Global EV Data 2024 | Energy Sector |
| Credit Card Fraud Detection | Isolation Forest Anomaly Detection | Kaggle Credit Card Transactions | Banking |
| NEPSE GFCLPO Stock (Linear) | Linear Regression → LSTM Deep Learning | GFCLPO NEPSE (2000-2021) | Nepal Finance |

## WHAT I BUILT & HOW

### 1. Bitcoin Price + Sentiment Analysis
- Loaded Bitcoin price and tweet datasets
- Applied TextBlob sentiment analysis on tweets
- Created LSTM neural network with 60-day sequences
- Combined price and sentiment features for prediction
- Built next-day Bitcoin price forecasting model

### 2. Breast Cancer Diagnosis
- Used sklearn breast cancer dataset
- Implemented Random Forest classifier with stratified split
- Generated confusion matrix and classification report
- Added feature importance analysis
- Created new patient prediction demonstration

### 3. Nepal Earthquake Damage Prediction
- Developed multi-class damage level prediction system
- Used features: magnitude, depth, buildings affected, population density
- Applied stratified train-test split for class balance
- Built Random Forest model with comprehensive evaluation
- Created real disaster scenario prediction pipeline

### 4. Global EV Sales Forecasting
- Processed IEA Global EV dataset (2024)
- Implemented Linear Regression, Polynomial Regression, and ARIMA models
- Performed ADF stationarity testing
- Generated 5-year business forecasts
- Compared performance across multiple forecasting techniques

### 5. Credit Card Fraud Detection
- Analyzed 284K transactions with 0.17% fraud rate
- Implemented Isolation Forest unsupervised anomaly detection
- Tuned contamination parameter for imbalanced data
- Built real-time fraud scoring pipeline
- Applied production banking evaluation metrics

### 6. NEPSE GFCLPO Linear Regression
- Processed 21 years of GFCLPO NEPSE data (2000-2021)
- Used multi-feature prediction with Close Price, Volume, High/Low
- Created forward-shifted target (Next_Close_Price)
- Calculated adjusted R-squared and directional accuracy
- Implemented stock market specific evaluation metrics

### 7. NEPSE GFCLPO LSTM
- Applied LSTM deep learning to same GFCLPO dataset
- Implemented dynamic sequence length handling
- Used MinMaxScaler for proper normalization
- Built production time series prediction pipeline
- Developed deep learning stock price forecasting system

## TECHNICAL SKILLS

Data Processing: pandas, numpy  
Visualization: matplotlib, seaborn, plotly.express  
Classical ML: scikit-learn (LinearRegression, RandomForest, IsolationForest)  
Time Series: statsmodels (ARIMA), ADF Stationarity Test  
Deep Learning: TensorFlow/Keras (LSTM)  
NLP: TextBlob sentiment analysis  
Preprocessing: StandardScaler, MinMaxScaler, PolynomialFeatures  
Evaluation: R-squared, RMSE, MAE, F1-Score, Confusion Matrix, Precision-Recall


