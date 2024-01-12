
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Veri setini yükleyinilmesi
df = pd.read_csv('/content/jobs_in_data.csv')

# Kategorik sütunların listesini oluşturulması
categorical_columns = ['job_title', 'job_category', 'employee_residence', 'experience_level',
                       'employment_type', 'work_setting', 'company_location', 'company_size', 'salary_currency']

# Kategorik sütunlar için one-hot encoding uygulanilması
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Hedef değişken ve özellikleri seçin
y = df_encoded['salary_in_usd']  # Hedef değişkeni güncellenilmesi
X = df_encoded.drop(['salary_in_usd'], axis=1)  # Hedef sütunu düşürülmesi

# Veriyi eğitim ve test setlerine ayırılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# XGBRegressor modelini tanımlanması
xgb_model = XGBRegressor(objective='reg:squarederror')

# Parametre gridini tanımlanması
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 500, 1000],
    'colsample_bytree': [0.3, 0.7]
}

# RandomizedSearchCV kullanarak hiperparametre optimizasyonu yapılması
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid,
                                   n_iter=10, cv=3, scoring='neg_mean_squared_error',
                                   n_jobs=-1, random_state=42)

# Eğitim veri seti üzerinde RandomizedSearchCV ile modeli eğitilmesi
random_search.fit(X_train, y_train)

# En iyi parametreleri gösterilmesi
print(f"En İyi Parametreler: {random_search.best_params_}")

# En iyi modeli alınması
best_xgb_model = random_search.best_estimator_

# Test seti üzerinde tahmin yap ve performansı değerlendirilmesi
predictions = best_xgb_model.predict(X_test)


# Verileri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ANN modelini oluşturma
model = Sequential()
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))  # Giriş katmanı
model.add(Dense(64, activation='relu'))  # Ara katman
model.add(Dense(1, activation='linear'))  # Çıkış katmanı

# Modeli derlemesi
model.compile(loss='mean_squared_error', optimizer='adam')

# Modeli eğitim veri seti üzerinde eğitmesi
model.fit(X_train_scaled, y_train, epochs=50, batch_size=10)

# Test veri seti üzerinde modelin performansını değerlendirmesi
loss = model.evaluate(X_test_scaled, y_test)
print(f'Test seti üzerindeki kayıp: {loss}')

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# XGBoost modeli için tahminler yapılması
xgb_predictions = best_xgb_model.predict(X_test)

# Hata metriklerini hesaplanılması
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_mae = mean_absolute_error(y_test, xgb_predictions)

print(f"XGBoost RMSE: {xgb_rmse}")
print(f"XGBoost MAE: {xgb_mae}")


# ANN modelini eğittikten sonra:
ann_predictions = model.predict(X_test)
ann_rmse = np.sqrt(mean_squared_error(y_test, ann_predictions))
ann_mae = mean_absolute_error(y_test, ann_predictions)
print(f"ANN RMSE: {ann_rmse}")
print(f"ANN MAE: {ann_mae}")