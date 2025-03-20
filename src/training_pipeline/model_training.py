import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.impute import SimpleImputer
import joblib
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam

def load_data(file_path):
    """
    Charge les données enrichies depuis un fichier CSV.
    """
    return pd.read_csv(file_path)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Entraîne et évalue un modèle, retourne les métriques.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calcul des métriques
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "explained_variance": explained_var,
        "r2": r2
    }

def create_bilstm_model(input_shape):
    """
    Crée un modèle BiLSTM.
    """
    model = Sequential([
        Bidirectional(LSTM(64, activation='relu'), input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def train_and_compare_models(data_path):
    """
    Entraîne et compare plusieurs modèles, sauvegarde le meilleur dans MLflow.
    """
    # Charger les données
    print(f"Chargement des données depuis {data_path}...")
    data = load_data(data_path)

    # Séparer les features (X) et la cible (y)
    X = data[['CO2', 'Humedad', 'PM2.5', 'Temperatura']]
    y = data['health_score']

  # Diviser les données en ensembles d'entraînement et de test
    train_size = int(0.8 * len(data))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Normalisation pour BiLSTM
    X_train_dl = X_train.to_numpy().reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_dl = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))

    # Modèle Random Forest
    print("Validation croisée avec Random Forest...")
    rf_model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
 # Optimisation des hyperparamètres avec GridSearchCV
    print("Optimisation des hyperparamètres avec GridSearchCV...")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")
    best_rf_model = grid_search.best_estimator_
    rf_metrics = evaluate_model(best_rf_model, X_train, X_test, y_train, y_test)

    # Modèle XGBoost
    print("Entraînement avec XGBoost...")
    xgb_model = xgb.XGBRegressor()
    xgb_metrics = evaluate_model(xgb_model, X_train, X_test, y_train, y_test)

    # Modèle BiLSTM
    print("Entraînement avec BiLSTM...")
    bilstm_model = create_bilstm_model((X_train_dl.shape[1], X_train_dl.shape[2]))
    bilstm_model.fit(X_train_dl, y_train, epochs=10, batch_size=32, verbose=0)
    bilstm_predictions = bilstm_model.predict(X_test_dl).flatten()
    bilstm_metrics = {
        "mse": mean_squared_error(y_test, bilstm_predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, bilstm_predictions)),
        "mae": mean_absolute_error(y_test, bilstm_predictions),
        "explained_variance": explained_variance_score(y_test, bilstm_predictions),
        "r2": r2_score(y_test, bilstm_predictions)
    }

    # Comparaison des modèles
    print("\nComparaison des modèles...")
    models_metrics = {
        "Random Forest": rf_metrics,
        "XGBoost": xgb_metrics,
        "BiLSTM": bilstm_metrics
    }
    best_model_name = max(models_metrics, key=lambda name: models_metrics[name]['r2'])
    best_model_metrics = models_metrics[best_model_name]

    print(f"Meilleur modèle : {best_model_name} avec R² = {best_model_metrics['r2']:.2f}")
    return models_metrics, best_model_name, best_rf_model, xgb_model, bilstm_model

mlflow.set_tracking_uri("../mlruns")  # Chemin vers le répertoire MLflow

def log_model_to_mlflow(models_metrics, best_model_name, best_rf_model, xgb_model, bilstm_model):
    """
    Enregistre les métriques et le meilleur modèle dans MLflow.
    """
    mlflow.set_experiment("best_model")  # Définit ou crée une expérience appelée "default_experiment"

    with mlflow.start_run():
        for model_name, metrics in models_metrics.items():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)

        if best_model_name == "Random Forest":
            mlflow.sklearn.log_model(best_rf_model, "best_model")
        elif best_model_name == "XGBoost":
            mlflow.sklearn.log_model(xgb_model, "best_model")
        elif best_model_name == "BiLSTM":
            bilstm_model.save("best_model")
            mlflow.log_artifact("best_model")

        print(f"Meilleur modèle ({best_model_name}) enregistré avec succès dans MLflow.")

if __name__ == "__main__":
    data_csv = "../../data/processed/final_enriched_data.csv"
    models_metrics, best_model_name, best_rf_model, xgb_model, bilstm_model = train_and_compare_models(data_csv)
    log_model_to_mlflow(models_metrics, best_model_name, best_rf_model, xgb_model, bilstm_model)