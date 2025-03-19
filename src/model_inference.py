import pandas as pd
import joblib
import mlflow.pyfunc
def load_data(file_path):
    """
    Charge les données d'entrée depuis un fichier CSV.
    """
    print(f"Chargement des données depuis {file_path}...")
    return pd.read_csv(file_path)

def filter_data_by_date_range(data, start_date, end_date):
    """
    Filtre les données en fonction d'un intervalle de dates (sans l'heure).
    """
    print("Filtrage des données par intervalle de dates...")
    # Convertir la colonne time_stamp en type datetime
    data['time_stamp'] = pd.to_datetime(data['time_stamp'])
    # Filtrer les données entre start_date et end_date
    filtered_data = data[(data['time_stamp'].dt.date >= pd.to_datetime(start_date).date()) &
                         (data['time_stamp'].dt.date <= pd.to_datetime(end_date).date())]
    print(f"{len(filtered_data)} lignes sélectionnées après filtrage.")
    return filtered_data

def load_model(model_uri):
    """
    Charge un modèle sauvegardé dans MLflow à partir de son URI.
    """
    print(f"Chargement du modèle depuis {model_uri}...")
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def make_predictions(model, data):
    """
    Effectue des prédictions avec le modèle chargé.
    """
    print("Génération des prédictions...")
    predictions = model.predict(data)
    return predictions

def save_predictions(predictions, original_data, output_path):
    """
    Sauvegarde les prédictions dans un fichier CSV avec les colonnes time_stamp et ID.
    """
    print(f"Sauvegarde des prédictions dans {output_path}...")
    # Créer un DataFrame avec les colonnes time_stamp, ID et les prédictions
    results = original_data[['time_stamp', 'ID']].copy()
    results['health_score_prediction'] = predictions
    results.to_csv(output_path, index=False)
    print("Prédictions sauvegardées avec succès.")

if __name__ == "__main__":
    # Chemin vers les données d'entrée
    input_csv = "../data/processed/intermediate_data.csv"  
    output_csv = "../data/processed/inference_predictions.csv"  # Chemin vers le fichier de sortie

    # URI du modèle sauvegardé dans MLflow
    model_uri = "runs:/3406c450a5a744dca29c8fc272a1da2b/best_model"
    # Intervalle de dates pour effectuer des prédictions
    start_date = "2024-06-20"
    end_date = "2024-06-28"

    # Charger les données
    data = load_data(input_csv)

    # Filtrer les données par intervalle de dates
    filtered_data = filter_data_by_date_range(data, start_date, end_date)

    # Sélectionner les colonnes nécessaires pour l'inférence
    features = filtered_data[['CO2', 'Humedad', 'PM2.5', 'Temperatura']]

    # Charger le modèle
    model = load_model(model_uri)

    # Effectuer les prédictions
    predictions = model.predict(features)

    # Sauvegarder les prédictions avec time_stamp et ID
    save_predictions(predictions, filtered_data, output_csv)