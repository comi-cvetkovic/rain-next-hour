# RainNextHour — Will it rain in the next hour?

## 0) Project name and group members
**Project:** RainNextHour  
**Group members:** Mihailo Cvetkovic

## 1) Dynamic data sources (no Kaggle static datasets)
We use **dynamic live weather data** from Open-Meteo:
- **Open-Meteo Geocoding API**: converts a city name into latitude/longitude.
- **Open-Meteo Forecast API**: provides hourly weather measurements for the past days and near-future.

The data is retrieved **on-demand** and updates continuously, ensuring the system always operates on fresh, real-world data rather than static datasets.

## 2) Prediction problem
We solve a **binary classification problem**:

- **Question:** Will it rain in the next hour?
- **Label definition:**  
  `rain_next_hour = 1` if precipitation in the next hour > 0.1 mm, otherwise `0`
- **Prediction horizon:** 1 hour ahead
- **Features (hour t):**
  - temperature (2m)
  - relative humidity
  - surface pressure
  - cloud cover
  - wind speed (10m)
  - current precipitation
- **Model:** Logistic Regression with feature scaling (`StandardScaler`) and `class_weight=balanced`
- **Evaluation metrics:** Classification report and ROC-AUC

## 3) How the project works (end-to-end)
The project is implemented as a simple end-to-end machine learning pipeline:

1. **Data ingestion**  
   Live hourly weather data is fetched from the Open-Meteo API for a given geographic location. For training, the model uses recent historical data (past days). For inference, the latest available hour is used.

2. **Feature engineering and labeling**  
   The hourly time series is transformed into a supervised learning dataset.  
   Weather conditions at time *t* are used as input features, while precipitation at time *t+1* is used to construct the binary target variable (rain or no rain).

3. **Model training**  
   A baseline Logistic Regression classifier is trained using scikit-learn.  
   Feature scaling is applied to ensure stable optimization, and class imbalance is handled by weighting the rain and no-rain classes.

4. **Model persistence and inference**  
   The trained model and metadata are saved as a `.joblib` file. During inference, the model is loaded without retraining and used to predict the probability of rain in the next hour.

5. **User interface**  
   A Streamlit web application allows users to input a city name, fetch live weather data, and receive a prediction along with contextual information such as recent precipitation history.

This design avoids training–serving skew by reusing the same feature definitions during both training and inference.

## 4) UI you provide to show value
We provide a **Streamlit-based web UI**:
- User inputs a city name (and optional country code)
- The app resolves the location and fetches live weather data
- The trained model predicts the probability of rain in the next hour
- The UI displays:
  - probability of rain
  - qualitative interpretation (likely / unlikely)
  - the latest feature values used for prediction
  - a table and line chart showing recent precipitation history

## 5) Technologies used
- **Programming language:** Python 3.x
- **Data ingestion:** `requests` + Open-Meteo APIs
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Model persistence:** joblib
- **UI:** Streamlit

## How to run the project
### 1) Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
