import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Bidirectional # type: ignore
from keras.optimizers import Adam # type: ignore

# Chargement des données
df = pd.read_csv('../data/processed/final_enriched_data.csv')

# Séparation des features et de la cible
X = df[['CO2', 'Humedad', 'PM2.5', 'Temperatura']]
y = df['health_score']

# Division temporelle : 80% pour l'entraînement, 20% pour le test
train_size = int(0.8 * len(df))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape des données pour les modèles LSTM et BiLSTM
X_train_dl = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_dl = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Modèles de Machine Learning
models = {
    'Random Forest': RandomForestRegressor(),
    'K-Neighbors Regressor': KNeighborsRegressor(),
    'XGBoost': xgb.XGBRegressor()
}

# Modèle LSTM
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Modèle BiLSTM
def create_bilstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, activation='relu'), input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Entraînement et évaluation des modèles
results = {}

# Modèles de Machine Learning
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    evs = explained_variance_score(y_test, predictions)
    results[name] = {'MSE': mse, 'MAE': mae, 'R2 Score': r2, 'Explained Variance': evs}

# Modèle LSTM
lstm_model = create_lstm_model((X_train_dl.shape[1], X_train_dl.shape[2]))
lstm_model.fit(X_train_dl, y_train, epochs=10, batch_size=32, verbose=0)
lstm_predictions = lstm_model.predict(X_test_dl).flatten()
results['LSTM'] = {
    'MSE': mean_squared_error(y_test, lstm_predictions),
    'MAE': mean_absolute_error(y_test, lstm_predictions),
    'R2 Score': r2_score(y_test, lstm_predictions),
    'Explained Variance': explained_variance_score(y_test, lstm_predictions)
}

# Modèle BiLSTM
bilstm_model = create_bilstm_model((X_train_dl.shape[1], X_train_dl.shape[2]))
bilstm_model.fit(X_train_dl, y_train, epochs=10, batch_size=32, verbose=0)
bilstm_predictions = bilstm_model.predict(X_test_dl).flatten()
results['BiLSTM'] = {
    'MSE': mean_squared_error(y_test, bilstm_predictions),
    'MAE': mean_absolute_error(y_test, bilstm_predictions),
    'R2 Score': r2_score(y_test, bilstm_predictions),
    'Explained Variance': explained_variance_score(y_test, bilstm_predictions)
}

# Affichage des résultats
print("\nRésultats des modèles :")
for model_name, metrics in results.items():
    print(f"\nModèle : {model_name}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")