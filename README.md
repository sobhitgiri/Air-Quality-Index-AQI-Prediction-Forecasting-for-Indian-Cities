# Air Quality Index (AQI) Prediction & Forecasting for Indian Cities

## 📌 Problem Statement
Air pollution is a critical health hazard in Indian cities.  
This project analyzes historical air quality data to:
1. Perform exploratory data analysis (EDA).
2. Predict AQI values (regression).
3. Classify air quality categories (Good, Moderate, Poor, etc.).
4. (Future milestone) Forecast AQI for 7 days ahead.

## 📊 Dataset
- **Source:** [Air Quality Data in India (2015–2024)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)  
- Includes pollutants (PM2.5, PM10, NO₂, SO₂, CO, O₃) and AQI across Indian cities.

## 🚀 Workflow Plan
1. **Week 1 (30%) – EDA**  
   - Load dataset, handle missing values, visualize trends, correlations, and seasonality.  
2. **Week 2 (60%) – Modeling**  
   - Train baseline models (Regression & Classification).  
   - Evaluate with RMSE/MAE and F1-score.  
3. **Week 3/Final (100%)**  
   - Forecast AQI (Prophet/LSTM).  
   - Deployment with Streamlit dashboard.  
   - Submit final report & PPT.

## ✅ Week 1 Deliverables
- `README.md`  
- `notebooks/data exploration.ipynb`
  
  ## 📌 Week-2 Progress

### 🔹 Tasks Completed
- Preprocessed dataset:
  - Handled missing values using forward fill  
  - Engineered new features (year, month, weekday, season)  
  - Encoded categorical variable (`city`) into numerical format  
  - Scaled numerical features using `StandardScaler`  
- Split dataset into training & testing sets (80:20)  
- Implemented baseline ML models:
  - **Linear Regression**  
  - **Random Forest Regressor**  
- Evaluated models using performance metrics:
  - RMSE (Root Mean Squared Error)  
  - MAE (Mean Absolute Error)  
  - R² Score (Coefficient of Determination)  

### 🔹 Deliverables
- Notebook: 02_preprocessing
            03_model_training
            04_evaluation
- Dataset: `air_quality_data.csv` (stored in repo as instructed)  
- Results: Model comparison table & performance metrics  

