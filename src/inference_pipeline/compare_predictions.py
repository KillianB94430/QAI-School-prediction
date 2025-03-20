import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.tracking import MlflowClient
import sys

# === 1️⃣ Charger les Données ===
real_data_path = "../../data/processed/final_enriched_data.csv"
pred_data_path = "../../data/processed/inference_predictions.csv"

df_real = pd.read_csv(real_data_path)
df_pred = pd.read_csv(pred_data_path)

# Convertir time_stamp en format datetime
df_real["time_stamp"] = pd.to_datetime(df_real["time_stamp"])
df_pred["time_stamp"] = pd.to_datetime(df_pred["time_stamp"])

# === 2️⃣ Sélectionner une période spécifique ===
def select_date_range(df, start_date, end_date):
    """Filtrer les données selon une plage de dates."""
    return df[(df["time_stamp"] >= start_date) & (df["time_stamp"] <= end_date)]

# Récupération des arguments passés par main.py
if len(sys.argv) >= 3:
    start_date = sys.argv[1]
    end_date = sys.argv[2]
else:
    raise ValueError("⚠️ Les dates de début et de fin sont requises en argument.")


start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filtrer les datasets
df_real_filtered = select_date_range(df_real, start_date, end_date)
df_pred_filtered = select_date_range(df_pred, start_date, end_date)

# === 3️⃣ Fusion des Données sur 'time_stamp' et 'ID' ===
df_compare = df_real_filtered.merge(df_pred_filtered, on=["time_stamp", "ID"], how="inner")

# Vérifier qu'on a bien des données après filtrage
if df_compare.empty:
    print("❌ Aucune donnée disponible pour cette période. Vérifiez vos dates !")
    exit()

# === 4️⃣ Calcul des Erreurs ===
df_compare["error"] = abs(df_compare["health_score"] - df_compare["health_score_prediction"])

# Métriques globales
mae = mean_absolute_error(df_compare["health_score"], df_compare["health_score_prediction"])
rmse = np.sqrt(mean_squared_error(df_compare["health_score"], df_compare["health_score_prediction"]))
r2 = r2_score(df_compare["health_score"], df_compare["health_score_prediction"])

print("\n📊 **Évaluation du Modèle sur la période sélectionnée**")
print(f"➡️ MAE (Mean Absolute Error) : {mae:.2f}")
print(f"➡️ RMSE (Root Mean Squared Error) : {rmse:.2f}")
print(f"➡️ R² Score : {r2:.2f}")

# === 5️⃣ Système de Monitoring ===
ALERT_THRESHOLD = 0.05  # Seuil de 5% de dégradation tolérée
mlflow.set_tracking_uri("../mlruns")

def get_last_run_metrics(experiment_name="best_model"):
    """
    Récupère les métriques du dernier run dans une expérience MLflow donnée.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"L'expérience '{experiment_name}' n'existe pas.")
    
    # Récupérer la dernière exécution de l'expérience
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"Aucune exécution trouvée pour l'expérience '{experiment_name}'.")
    
    last_run = runs[0]
    # Récupérer toutes les métriques du dernier run
    return last_run.data.metrics

def monitor_model_performance(new_metrics):
    """
    Compare les nouvelles métriques avec celles du dernier run MLflow.
    """
    last_metrics = get_last_run_metrics()
    alert = False
    print("\n📊 Comparaison des performances du modèle :")
    for metric, last_value in last_metrics.items():
        new_value = new_metrics.get(metric, None)
        if new_value is not None:
            degradation = (last_value - new_value) / last_value if last_value != 0 else 0
            print(f"{metric}: Last = {last_value:.4f}, New = {new_value:.4f}, Degradation = {degradation:.2%}")
            if degradation > ALERT_THRESHOLD:
                alert = True
                print(f"⚠️ ALERTE: Dégradation significative détectée sur {metric}!")
    if not alert:
        print("✅ Le modèle maintient ses performances. Pas d'alerte.")

# Exécuter le monitoring
new_metrics = {"mse": mean_squared_error(df_compare["health_score"], df_compare["health_score_prediction"]),
               "rmse": rmse,
               "mae": mae,
               "r2": r2}
monitor_model_performance(new_metrics)

# === 6️⃣ Visualisation des Résultats ===
## 📍 Comparaison Prédictions vs Valeurs Réelles
plt.figure(figsize=(10, 6))
plt.scatter(df_compare["health_score"], df_compare["health_score_prediction"], alpha=0.5)
plt.plot([df_compare["health_score"].min(), df_compare["health_score"].max()],
         [df_compare["health_score"].min(), df_compare["health_score"].max()],
         color="red", linestyle="--")
plt.xlabel("Vérité Terrain (health_score)")
plt.ylabel("Prédictions (health_score_prediction)")
plt.title(f"Comparaison des Prédictions vs Vérités Terrain\n({start_date.date()} → {end_date.date()})")
plt.show()