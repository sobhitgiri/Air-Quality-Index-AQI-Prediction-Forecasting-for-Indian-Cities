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

🚀 Features:-
📊 Data Exploration: Visualize AQI data, distributions, and correlations
🧹 Preprocessing: Handle missing values, remove outliers, and select features
🤖 Modeling: Train Random Forest, Logistic Regression, or XGBoost models
🏆 Evaluation: View classification reports, confusion matrices, and feature importances
🔮 Prediction: Instantly predict AQI category for new data
💾 Model Saving/Loading: Save and reload trained models
🧪 Testing: Automated tests for data and edge cases
🎨 Modern UI: Clean, responsive, and user-friendly interface

<img width="1919" height="1004" alt="Screenshot 2025-09-13 214312" src="https://github.com/user-attachments/assets/d0ad16ac-d3ac-4b4c-b11a-d24e91627267" />

<img width="827" height="875" alt="Screenshot 2025-09-13 214437" src="https://github.com/user-attachments/assets/e10d0f2f-da2d-4e9d-9a40-9e8f69b3a949" />

<img width="822" height="708" alt="Screenshot 2025-09-13 214503" src="https://github.com/user-attachments/assets/05edc1ac-77f4-4036-86bd-369dadb2f751" />

<img width="825" height="629" alt="Screenshot 2025-09-13 214520" src="https://github.com/user-attachments/assets/2d45ca02-fba0-4997-abde-0dd020745879" />

<img width="588" height="698" alt="Screenshot 2025-09-13 214552" src="https://github.com/user-attachments/assets/5cae533b-0a8a-44f0-857a-4395bfc0ec2b" />

<img width="1520" height="869" alt="Screenshot 2025-09-13 214644" src="https://github.com/user-attachments/assets/9ef88dc0-783f-4691-8d48-d0db83f9ca91" />

<img width="1177" height="872" alt="Screenshot 2025-09-13 214825" src="https://github.com/user-attachments/assets/7857011a-c87b-4ce2-b635-1a6703ba2673" />

<img width="1518" height="874" alt="Screenshot 2025-09-13 214856" src="https://github.com/user-attachments/assets/ccff6b81-aa6f-4442-ac5f-2c25c1d6714c" />

<img width="1522" height="819" alt="Screenshot 2025-09-13 215001" src="https://github.com/user-attachments/assets/07d3373a-c9b2-45d5-8d54-69ca5672648e" />

<img width="1519" height="871" alt="Screenshot 2025-09-13 215106" src="https://github.com/user-attachments/assets/cd36486a-9767-42c3-b472-293e49a77b79" />

<img width="1522" height="874" alt="Screenshot 2025-09-13 215259" src="https://github.com/user-attachments/assets/4c571603-0d08-48bb-998d-c6cd241d132f" />

<img width="1520" height="782" alt="Screenshot 2025-09-13 215422" src="https://github.com/user-attachments/assets/7824c10c-dd05-4d6e-93d9-34467b7c515f" />

<img width="401" height="70" alt="Screenshot 2025-09-13 215452" src="https://github.com/user-attachments/assets/1f911cf9-9279-47ef-bfe3-808ffdf5264d" />
<img width="1306" height="823" alt="Screenshot 2025-09-13 233138" src="https://github.com/user-attachments/assets/6cbf8903-7d17-4583-b28a-84a31ee6d472" />

<img width="345" height="69" alt="Screenshot 2025-09-13 215532" src="https://github.com/user-attachments/assets/1fd0a8a5-e280-4aa1-bc78-cf695a298e53" />
<img width="1120" height="854" alt="Screenshot 2025-09-13 215558" src="https://github.com/user-attachments/assets/dae99430-cf84-4155-88fb-2f50fd231e6d" />

🧑‍� Usage:-
   - Use the sidebar to upload data, select preprocessing, features, and model
   - Train and evaluate the dataset and provide output
   - Predict AQI for new data instantly
   - Visualize results and download models

🧐 Data Validation:-
   - Built-in checks for missing values, class balance, and outliers
   - Outlier visualization and feature importance explanations included

👤 Author:-
 - Sobhit Giri

📄 License:-
This project is licensed under the MIT License - [MIT License](https://opensource.org/licenses/MIT)

 











