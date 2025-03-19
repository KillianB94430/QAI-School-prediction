import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

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

# Modèles de Machine Learning
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regressor': SVR(),
    'K-Neighbors Regressor': KNeighborsRegressor(),
    'XGBoost': xgb.XGBRegressor()
}

results = {}

# Entraînement et évaluation des modèles de Machine Learning
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    evs = explained_variance_score(y_test, predictions)
    results[name] = {'MSE': mse, 'MAE': mae, 'R2 Score': r2, 'Explained Variance': evs}

# Affichage des résultats
print("\nRésultats des modèles de Machine Learning :")
for model_name, metrics in results.items():
    print(f"\nModèle : {model_name}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")