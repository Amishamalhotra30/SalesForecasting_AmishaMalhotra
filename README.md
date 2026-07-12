# SalesForecasting_AmishaMalhotra
# 📈 End-to-End Sales Forecasting & Demand Intelligence System

An end-to-end machine learning project for retail sales analysis, demand forecasting, anomaly detection, and product demand segmentation using the Superstore Sales dataset. The project includes exploratory data analysis, multiple forecasting models, anomaly detection techniques, customer-friendly visualizations, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

Retail businesses rely on accurate demand forecasting to optimize inventory, improve supply chain efficiency, and support business decision-making. This project develops a complete sales forecasting and demand intelligence system that combines machine learning, time-series analysis, anomaly detection, and clustering techniques into a single interactive dashboard.

The system enables users to:

- Analyze historical sales trends
- Forecast future sales using multiple models
- Detect unusual sales behavior
- Segment products based on demand characteristics
- Visualize insights through an interactive Streamlit dashboard

---

## 🚀 Features

### 📊 Exploratory Data Analysis
- Data preprocessing and cleaning
- Monthly and yearly sales analysis
- Category-wise and region-wise sales analysis
- Time-series visualization

### 📈 Time-Series Forecasting
Three forecasting models were implemented and compared:

- SARIMA
- Prophet
- XGBoost (Best Performing Model)

Model comparison was performed using:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

---

### 🚨 Anomaly Detection

Two anomaly detection techniques were implemented:

- Isolation Forest
- Z-Score Analysis

These methods identify unusual weekly sales behavior caused by promotions, seasonal demand, or unexpected business events.

---

### 📦 Product Demand Segmentation

Products were segmented using K-Means Clustering based on:

- Total Sales
- Sales Growth Rate
- Sales Volatility
- Average Order Value

Principal Component Analysis (PCA) was used to visualize demand clusters.

---

### 🌐 Interactive Streamlit Dashboard

The dashboard contains four interactive pages:

- 📊 Sales Overview
- 📈 Forecast Explorer
- 🚨 Anomaly Report
- 📦 Product Demand Segments

---

## 📂 Project Structure

```
SalesForecasting_AmishaMalhotra/
│
├── analysis.ipynb
├── app.py
├── requirements.txt
├── train.csv
├── summary.docx
├── charts/
│   ├── monthly_sales_trend.png
│   ├── time_series_decomposition.png
│   ├── differenced_series.png
│   ├── sarima_forecast.png
│   ├── prophet_forecast.png
│   ├── xgboost_forecast.png
│   ├── category_region_forecast.png
│   ├── isolation_forest.png
│   ├── anomaly_comparison.png
│   ├── elbow_method.png
│   └── product_clusters.png
```

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- Statsmodels
- Prophet
- XGBoost
- Streamlit

---

## 📈 Model Performance

| Model | MAE | RMSE | MAPE |
|------|------------:|------------:|------------:|
| SARIMA | 19,244.49 | 19,950.07 | 20.53% |
| Prophet | 20,250.79 | 22,318.41 | 21.86% |
| **XGBoost** | **15,398.11** | **19,461.74** | **14.99%** |

**Best Model:** XGBoost

---

## ▶️ Running the Project Locally

Clone the repository:

```bash
git clone https://github.com/<your-username>/SalesForecasting_AmishaMalhotra.git
```

Navigate to the project folder:

```bash
cd SalesForecasting_AmishaMalhotra
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

---
## Live Demo

**Streamlit App:** https://salesforecastingamishamalhotra-ocst5nxj9nuztsxgbqearh.streamlit.app/

## 📋 Business Recommendations

- Deploy the XGBoost model for operational sales forecasting.
- Continuously monitor anomalies to identify unusual sales behavior.
- Use product demand segmentation to optimize inventory planning and replenishment strategies.
- Leverage forecast insights to improve procurement and warehouse management.

---

## 📌 Future Enhancements

- Incorporate external variables such as holidays, promotions, and economic indicators.
- Extend forecasting to longer prediction horizons.
- Add deep learning models such as LSTM or Temporal Fusion Transformer.
- Enable real-time forecasting using live sales data.
- Integrate cloud-based deployment with automated model retraining.

---

## 👩‍💻 Author

**Amisha Malhotra**

B.Tech Computer Science & Engineering

Graphic Era Deemed to be University

---

## 📄 License

This project is developed for academic and educational purposes.
