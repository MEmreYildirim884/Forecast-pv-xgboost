import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os

# 1. LOAD DATA
try:
    df = pd.read_csv('b.csv')
    # Rename the column to match the expected 'pv_production' or just change usage.
    # Changing usage is better.
    # The snippet shows column name is 'consumption'.
    # Let's just rename it to pv_production to minimize code changes or update the code to use 'consumption'.
    # Updating code to use 'consumption' is cleaner but renaming is safer for "don't change anything else" requests.
    # However, the user asked to "do the same with the new file".
    # I will stick to the user's logic but adapt the column name variable.
    
    # Actually, simply renaming the column in the dataframe after load is the least intrusive change.
    df.columns = ['pv_production'] # Renaming 'consumption' to 'pv_production' for the code to work as is.
    
    total_days = len(df) // 24
    print(f"Total rows: {len(df)}")
    print(f"Total days: {total_days}")
except FileNotFoundError:
    print("HATA: 'b.csv' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    dates = pd.date_range(start='2024-01-01', periods=240, freq='h')
    df = pd.DataFrame({'pv_production': np.abs(np.sin(np.linspace(0, 100, 240))) * 10}, index=dates)
    total_days = 10

# 2. CREATE FEATURE EXTRACTION FUNCTION
def extract_features_from_day(day_data):
    """Extract features from a single day (24 hours)"""
    if len(day_data) != 24:
        raise ValueError(f"Need 24 hours, got {len(day_data)}")
    
    features = []
    
    # 1. Basic stats
    features.append(np.mean(day_data))
    features.append(np.std(day_data))
    features.append(np.max(day_data))
    features.append(np.min(day_data))
    features.append(np.median(day_data))
    
    # 2. Trend
    features.append(np.polyfit(range(24), day_data, 1)[0])
    
    # 3. Time segments
    features.append(np.mean(day_data[6:12]))   # morning 6-12
    features.append(np.mean(day_data[12:16]))  # noon 12-16
    features.append(np.mean(day_data[16:20]))  # evening 16-20
    
    # 4. Lag features
    features.append(day_data[-1])   # last hour
    features.append(day_data[-2])   # 2nd last
    features.append(day_data[-3])   # 3rd last
    
    # 5. Differences
    features.append(day_data[-1] - day_data[-2])
    features.append(day_data[-2] - day_data[-3])
    
    # 6. Moving averages
    features.append(np.mean(day_data[-3:]))
    features.append(np.mean(day_data[-6:]))
    
    return np.array(features).reshape(1, -1)

# 3. PREPARE TRAINING DATA (use all days except last one for training)
def prepare_training_data(data):
    """Prepare training data: each day predicts next day"""
    X_train = []
    y_train = []
    
    # Her gün için: o gün -> sonraki gün (son gün hariç)
    for day in range(total_days - 2):  # -2 çünkü son günün hedefi yok ve day+1 olacak
        # Current day
        start_idx = day * 24
        end_idx = (day + 1) * 24
        current_day = data[start_idx:end_idx]
        
        # Next day (target)
        next_start = (day + 1) * 24
        next_end = (day + 2) * 24
        next_day = data[next_start:next_end]
        
        # Extract features from current day
        features = extract_features_from_day(current_day).flatten()
        
        X_train.append(features)
        y_train.append(next_day)
    
    return np.array(X_train), np.array(y_train)

# 4. TRAIN MODELS
print("\n" + "="*60)
print("TRAINING MODELS ON ALL DAYS (except last)")
print("="*60)

# Get training data (all days except last)
train_data = df['pv_production'].values[:-24]  # All except last day
X_train, y_train = prepare_training_data(train_data)

print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Training on {X_train.shape[0]} day pairs")

# Train 24 models (one for each hour)
models = []

for hour in range(24):
    # print(f"Training model for hour {hour+1:02d}/24...", end=" ")
    
    # Target for this hour
    y_hour = y_train[:, hour]
    
    # XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    # Train
    model.fit(X_train, y_hour)
    models.append(model)
    
    # Quick performance check
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_hour, train_pred)
    # print(f"MAE: {train_mae:.4f}")

# 5. MAKE PREDICTIONS FOR ALL DAYS (Day 2 to Last Day)
print("\n" + "="*60)
print("MAKING PREDICTIONS FOR ALL DAYS")
print("="*60)

# Get actual data for all days
actuals = {}
all_day_data = df['pv_production'].values

for day in range(1, total_days + 1):
    start_idx = (day-1) * 24
    end_idx = day * 24
    actuals[f'Day {day}'] = all_day_data[start_idx:end_idx]

# Make predictions for each day (starting from Day 2)
predictions = {}

# Day 1'den Day 2'yi tahmin et
day1_data = actuals['Day 1']
day1_features = extract_features_from_day(day1_data)

day2_pred = []
for hour in range(24):
    pred = models[hour].predict(day1_features)[0]
    day2_pred.append(max(0, pred))  # No negative values
predictions['Day 2'] = np.array(day2_pred)

# Day 2'den Day 3'ü tahmin et
day2_data = actuals['Day 2']
day2_features = extract_features_from_day(day2_data)

day3_pred = []
for hour in range(24):
    pred = models[hour].predict(day2_features)[0]
    day3_pred.append(max(0, pred))
predictions['Day 3'] = np.array(day3_pred)

# Tüm günler için tahmin yap (recursive)
for day in range(3, total_days):
    # Previous day's actual data (for features)
    prev_day_data = actuals[f'Day {day}']
    prev_day_features = extract_features_from_day(prev_day_data)
    
    # Predict current day
    current_pred = []
    for hour in range(24):
        pred = models[hour].predict(prev_day_features)[0]
        current_pred.append(max(0, pred))
    
    predictions[f'Day {day+1}'] = np.array(current_pred)

print(f"Predictions made for Days 2 to {total_days}")

# 6. CALCULATE PERFORMANCE FOR ALL DAYS
print("\n" + "="*60)
print("PERFORMANCE METRICS FOR ALL DAYS")
print("="*60)

all_metrics = []

for day in range(2, total_days + 1):
    actual = actuals[f'Day {day}']
    predicted = predictions[f'Day {day}']
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    total_actual = np.sum(actual)
    total_pred = np.sum(predicted)
    error_pct = abs(total_actual - total_pred) / (total_actual + 1e-8) * 100
    
    all_metrics.append({
        'Day': day,
        'MAE': mae,
        'RMSE': rmse,
        'Total_Actual': total_actual,
        'Total_Predicted': total_pred,
        'Error_Percentage': error_pct
    })
    
    # print(f"\nDay {day:2d}: MAE={mae:6.2f} kW, RMSE={rmse:6.2f} kW, Error={error_pct:5.1f}%")

# --- BURAYA DİKKAT: İSTEDİĞİNİZ YÜZDELİK DOĞRULUK EKLENDİ ---
print("\n" + "*"*60)
# Ortalama hata yüzdesini tüm metriklerden al
mean_error_percentage = np.mean([m['Error_Percentage'] for m in all_metrics])
# Doğruluk oranı = 100 - Hata
accuracy = max(0, 100 - mean_error_percentage)
print(f"SİSTEMİN GENEL TAHMİN DOĞRULUĞU: %{accuracy:.2f}")
print("*"*60)