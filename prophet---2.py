import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import logging


# Gereksiz logları kapat
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

# ==========================================
# 1. VERİ YÜKLEME VE HAZIRLIK (b.csv)
# ==========================================
print("1. b.csv verisi yükleniyor ve hazırlanıyor...")

# Dosyayı oku
df = pd.read_csv('b.csv')

# Prophet formatına uyarla: Sütun ismini 'y' yap
df.columns = ['y']

# Veri saatlik olduğu için saatlik tarih (ds) oluşturuyoruz
# Not: b.csv 841 satır, bu yaklaşık 35 gün eder.
df['ds'] = pd.date_range(start='2024-01-01', periods=len(df), freq='h')

# --- KRİTİK HAMLE: REGRESSOR (ÖNCEKİ SAAT VERİSİ) ---
# Bir önceki saatin verisini "onceki_deger" olarak ekliyoruz.
df['onceki_deger'] = df['y'].shift(1)

# İlk satırda geçmiş veri olmadığı için siliyoruz
df = df.dropna().reset_index(drop=True)

print(f"Toplam Veri: {len(df)} saat")
print("-" * 50)

# ==========================================
# 2. MODEL EĞİTİMİ VE TAHMİN (Walk-Forward)
# ==========================================
predictions = []
actuals = []

print("2. Prophet ile Döngüsel Tahmin Başlıyor...")
print("(Bu işlem verinin boyutuna göre biraz zaman alabilir, lütfen bekleyin...)")

# Test için ilk 24 saati (1. günü) eğitim verisi olarak alıp 25. saatten başlıyoruz
start_index = 24 

# Hızlandırmak için her adımı değil, 24 saatlik blokları tahmin edebiliriz
# Ancak en yüksek hassasiyet için saat saat ilerliyoruz (Sizin şablonunuzdaki gibi)
for i in range(start_index, len(df)):
    
    # a) Eğitim Seti: Şu ana kadarki tüm veriler
    train_df = df.iloc[:i].copy()
    
    # b) Hedef: Sıradaki saatin gerçek değeri ve regressor değeri
    actual_value = df.iloc[i]['y']
    regressor_value = df.iloc[i]['onceki_deger']
    
    # c) Model Kurulumu (Regressor aktif)
    # Saatlik veri olduğu için daily_seasonality=True çok önemlidir.
    m = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False)
    m.add_regressor('onceki_deger')
    
    m.fit(train_df)
    
    # d) Gelecek (Sadece 1 saat sonrası)
    future = m.make_future_dataframe(periods=1, freq='h')
    
    # Gelecek veri setine bilinen 'onceki_deger'i ekle
    # (Son satır bizim tahmin edeceğimiz yerdir)
    future['onceki_deger'] = df.iloc[:i+1]['onceki_deger'].values
    
    # e) Tahmin
    forecast = m.predict(future)
    predicted_value = forecast.iloc[-1]['yhat']
    
    # Negatif sonuçları engelle
    predicted_value = max(0, predicted_value)
    
    predictions.append(predicted_value)
    actuals.append(actual_value)
    
    # İlerlemeyi göster (Her 50 saatte bir yazdır)
    if i % 50 == 0:
        diff = abs(actual_value - predicted_value)
        print(f"Saat {i}/{len(df)} | Gerçek: {actual_value:.2f} | Tahmin: {predicted_value:.2f}")

# ==========================================
# 3. SONUÇ VE RAPORLAMA
# ==========================================
print("\n" + "="*60)
print("SONUÇLAR")
print("="*60)

y_true = np.array(actuals)
y_pred = np.array(predictions)

mae = mean_absolute_error(y_true, y_pred)
total_actual = np.sum(y_true)
error_percentage = (np.sum(np.abs(y_true - y_pred)) / (total_actual + 1e-8)) * 100
accuracy = max(0, 100 - error_percentage)

print(f"Toplam Tahmin Edilen Saat: {len(y_true)}")
print(f"MAE: {mae:.2f}")
print(f"Hata Payı: %{error_percentage:.2f}")
print(f"DOĞRULUK ORANI: %{accuracy:.2f}")

# Grafik
plt.figure(figsize=(15, 6))
plt.plot(y_true, label='Gerçek Tüketim', color='black', alpha=0.7)
plt.plot(y_pred, label='Prophet Tahmini', color='green', linestyle='--')
plt.title(f'b.csv Prophet Tahmini - Doğruluk: %{accuracy:.2f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('b_csv_prophet_sonuc.png')
print("Grafik kaydedildi.")



#mae = mean absolute error