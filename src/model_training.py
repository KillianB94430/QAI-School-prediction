import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.impute import SimpleImputer
import joblib
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

def train_and_optimize_model(data_path):
    global best_rf_model  # Declare as global to make it accessible outside the function
    """
    Entraîne un modèle de machine learning avec validation croisée et optimisation des hyperparamètres.
    """
    # Charger les données
    print(f"Chargement des données depuis {data_path}...")
    data = load_data(data_path)

    # Séparer les features (X) et la cible (y)
    X = data[['CO2', 'Humedad', 'PM2.5', 'Temperatura']]
    y = data['health_score']

    # Gérer les valeurs manquantes avec SimpleImputer
    print("Imputation des valeurs manquantes...")
    imputer = SimpleImputer(strategy='mean')  # Remplir les NaN avec la moyenne
    X = imputer.fit_transform(X)

    # Diviser les données en ensembles d'entraînement et de test
    # Division temporelle : 80% pour l'entraînement, 20% pour le test
    train_size = int(0.8 * len(data_path))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Modèle de base : Random Forest
    print("Validation croisée avec Random Forest...")
    rf_model = RandomForestRegressor(random_state=42)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
    print(f"Scores de validation croisée (R²) : {cv_scores}")
    print(f"Score moyen (R²) : {cv_scores.mean():.2f}")

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

    # Évaluation du modèle optimisé
    print("Évaluation du modèle optimisé...")
    rf_metrics = evaluate_model(best_rf_model, X_train, X_test, y_train, y_test)
    print(f"Random Forest - MSE : {rf_metrics['mse']:.2f}, RMSE : {rf_metrics['rmse']:.2f}, "
          f"MAE : {rf_metrics['mae']:.2f}, Explained Variance : {rf_metrics['explained_variance']:.2f}, "
          f"R² : {rf_metrics['r2']:.2f}")

    # Comparaison avec un autre modèle : Régression Linéaire
    print("Comparaison avec un modèle de Régression Linéaire...")
    lr_model = LinearRegression()
    lr_metrics = evaluate_model(lr_model, X_train, X_test, y_train, y_test)
    print(f"Régression Linéaire - MSE : {lr_metrics['mse']:.2f}, RMSE : {lr_metrics['rmse']:.2f}, "
          f"MAE : {lr_metrics['mae']:.2f}, Explained Variance : {lr_metrics['explained_variance']:.2f}, "
          f"R² : {lr_metrics['r2']:.2f}")

    # Suivi avec MLflow
    with mlflow.start_run():
        # Enregistrer les métriques du modèle Random Forest optimisé
        for metric_name, metric_value in rf_metrics.items():
            mlflow.log_metric(f"rf_{metric_name}", metric_value)

        # Enregistrer les métriques du modèle de Régression Linéaire
        for metric_name, metric_value in lr_metrics.items():
            mlflow.log_metric(f"lr_{metric_name}", metric_value)

        # Enregistrer le meilleur modèle Random Forest
        mlflow.sklearn.log_model(best_rf_model, "optimized_random_forest_model")
        print("Modèle optimisé enregistré avec succès dans MLflow.")

if __name__ == "__main__":
    # Chemin vers les données enrichies
    data_csv = "../data/processed/final_enriched_data.csv"

    best_rf_model = train_and_optimize_model(data_csv)
    train_and_optimize_model(data_csv)
    # Sauvegarder le modèle optimisé localement
    model_save_path = "../models/optimized_random_forest_model.pkl"
    joblib.dump(best_rf_model, model_save_path)
    print(f"Modèle optimisé sauvegardé localement à l'emplacement : {model_save_path}")