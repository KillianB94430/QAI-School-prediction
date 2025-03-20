import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.model_selection import RandomizedSearchCV
import mlflow
import mlflow.sklearn
import numpy as np
import xgboost as xgb

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

def optimize_xgboost(X_train, y_train):
    """
    Effectue une recherche d'hyperparamètres pour XGBoost.
    """
    param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }

    xgb_model = xgb.XGBRegressor()
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=10,  # Nombre d'itérations pour la recherche
        scoring="r2",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train)

    print(f"Meilleurs hyperparamètres : {random_search.best_params_}")
    return random_search.best_estimator_

def train_and_evaluate_xgboost(data_path):
    """
    Entraîne et évalue le modèle XGBoost, puis sauvegarde le modèle et ses métriques dans MLflow.
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

    # Optimisation des hyperparamètres pour XGBoost
    print("Optimisation des hyperparamètres pour XGBoost...")
    best_xgb_model = optimize_xgboost(X_train, y_train)

    # Évaluation du modèle optimisé
    xgb_metrics = evaluate_model(best_xgb_model, X_train, X_test, y_train, y_test)

    print("\n📊 Résultats du modèle XGBoost optimisé :")
    print(f"R² : {xgb_metrics['r2']:.2f}")
    print(f"MSE : {xgb_metrics['mse']:.2f}")
    print(f"RMSE : {xgb_metrics['rmse']:.2f}")
    print(f"MAE : {xgb_metrics['mae']:.2f}")
    print(f"Explained Variance : {xgb_metrics['explained_variance']:.2f}")

    # Sauvegarder le modèle et les métriques dans MLflow
    log_xgboost_to_mlflow(best_xgb_model, xgb_metrics)

def log_xgboost_to_mlflow(model, metrics):
    """
    Enregistre le modèle XGBoost et ses métriques dans MLflow.
    """
    mlflow.set_tracking_uri("../mlruns")  # Chemin vers le répertoire MLflow
    mlflow.set_experiment("best_model")  # Définit ou crée une expérience appelée "best_model"

    with mlflow.start_run():
        # Enregistrer les métriques
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Enregistrer le modèle
        mlflow.sklearn.log_model(model, "best_model")

        print("✅ Modèle XGBoost optimisé enregistré avec succès dans MLflow.")

if __name__ == "__main__":
    data_csv = "../../data/processed/final_enriched_data.csv"
    train_and_evaluate_xgboost(data_csv)