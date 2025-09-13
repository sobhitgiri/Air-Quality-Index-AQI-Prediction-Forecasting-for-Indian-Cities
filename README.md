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
## 📌 Week 1 Progress (Milestone – 30%)

### 🔹 Tasks Completed
- Created GitHub repository and project structure  
- Added `README.md` with project title, dataset source, and workflow plan  
- Loaded dataset and performed initial checks for missing values  
- Conducted exploratory analysis with at least 3 visualizations:  
  - AQI/PM2.5 trend analysis  
  - Correlation heatmap between pollutants and AQI  
  - Seasonal/monthly AQI variation  

---

### 🔹 Deliverables
- **Files**:  
  - `README.md`  
  - `01_data_exploration.ipynb`  

- **Results**:  
  - Dataset successfully loaded  
  - Missing values identified and handled  
  - Visual insights generated for AQI trends, correlations, and seasonality  

  ## ✅ Week 2 Deliverables
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
- Notebook: 02_preprocessing.ipynb
            03_model_training.ipynb
            04_evaluation.ipynb
- Dataset: `air_quality_data.csv` (stored in repo as instructed)  
- Results: Model comparison table & performance metrics
  
 ## ✅ Week 3 Deliverables 
## 📌 Week 3 Progress (Final Milestone – 100%)

### 🔹 Tasks Completed

#### Advanced Model Training & Tuning
- Implemented **XGBoost Regressor** with hyperparameter tuning (`GridSearchCV`)  
- Retrained the model on the best parameters  
- Evaluated performance with **RMSE, MAE, and R² score**  
- Analyzed **feature importance** to identify key pollutants affecting AQI  

#### Model Saving
- Saved trained model as `models/aqi_model.pkl`  
- Saved fitted scaler as `models/scaler.pkl`  

#### Streamlit App Development
- Built an interactive **Streamlit app (`app.py`)** for real-time AQI prediction  
- Users can input pollutant values (**PM2.5, PM10, NO₂, SO₂, CO, O₃**)  
- Outputs predicted AQI value along with AQI **category** (Good, Moderate, Poor, etc.)  
- Integrated `config.py` for cleaner configuration management  

#### Deployment
- Added `requirements.txt` for dependency management  
- Deployed app on **Streamlit Cloud**  
- Accessible at: https://sobhitgiri-air-quality-index-aqi-prediction-for-week3app-c8zaoj.streamlit.app/

---

### 🔹 Deliverables
- **Notebooks**:  
  - `03_model_training.ipynb`  
  - `04_evaluation.ipynb`  

- **Deployment files**:  
  - `WEEK 3/app.py`  
  - `WEEK 3/config.py`  
  - `requirements.txt`  

- **Models**:  
  - `models/aqi_model.pkl`  
  - `models/scaler.pkl`  

- **Deployed Streamlit App**: https://sobhitgiri-air-quality-index-aqi-prediction-for-week3app-c8zaoj.streamlit.app/

  <img width="1300" height="551" alt="Screenshot 2025-09-13 232050" src="https://github.com/user-attachments/assets/7ed36ee6-9e34-4f3e-83d1-2c08e7383715" />

🚀 Features
📊 Data Exploration: Visualize AQI data, distributions, and correlations
🧹 Preprocessing: Handle missing values, remove outliers, and select features
🤖 Modeling: Train Random Forest, Logistic Regression, or XGBoost models
🏆 Evaluation: View classification reports, confusion matrices, and feature importances
🔮 Prediction: Instantly predict AQI category for new data
💾 Model Saving/Loading: Save and reload trained models
🧪 Testing: Automated tests for data and edge cases
🎨 Modern UI: Clean, responsive, and user-friendly interface

