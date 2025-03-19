import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === 1️⃣ Charger les Données ===
real_data_path = "../data/processed/final_enriched_data.csv"
pred_data_path = "../data/processed/inference_predictions.csv"

df_real = pd.read_csv(real_data_path)
df_pred = pd.read_csv(pred_data_path)

# Convertir time_stamp en format datetime
df_real["time_stamp"] = pd.to_datetime(df_real["time_stamp"])
df_pred["time_stamp"] = pd.to_datetime(df_pred["time_stamp"])

# === 2️⃣ Sélectionner une période spécifique ===
def select_date_range(df, start_date, end_date):
    """Filtrer les données selon une plage de dates."""
    return df[(df["time_stamp"] >= start_date) & (df["time_stamp"] <= end_date)]

# Saisie utilisateur pour la période
start_date = input("📆 Entrez la date de début (YYYY-MM-DD) : ")
end_date = input("📆 Entrez la date de fin (YYYY-MM-DD) : ")

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

# === 5️⃣ Visualisation des Résultats ===

## 📍 5.1 Comparaison Prédictions vs Valeurs Réelles
plt.figure(figsize=(10, 6))
plt.scatter(df_compare["health_score"], df_compare["health_score_prediction"], alpha=0.5)
plt.plot([df_compare["health_score"].min(), df_compare["health_score"].max()], 
         [df_compare["health_score"].min(), df_compare["health_score"].max()], 
         color="red", linestyle="--")  # Ligne parfaite
plt.xlabel("Vérité Terrain (health_score)")
plt.ylabel("Prédictions (health_score_prediction)")
plt.title(f"Comparaison des Prédictions vs Vérités Terrain\n({start_date.date()} → {end_date.date()})")
plt.show()

## 📍 5.2 Distribution des Erreurs
plt.figure(figsize=(10, 6))
plt.hist(df_compare["error"], bins=50, color="blue", alpha=0.7)
plt.xlabel("Erreur Absolue (|Prédiction - Vérité|)")
plt.ylabel("Nombre d'échantillons")
plt.title(f"Distribution des Erreurs de Prédiction\n({start_date.date()} → {end_date.date()})")
plt.show()

## 📍 5.3 Analyse Temporelle des Prédictions
sensor_id = df_compare["ID"].iloc[0]  # Prendre un capteur au hasard
df_filtered = df_compare[df_compare["ID"] == sensor_id]

plt.figure(figsize=(12, 6))
plt.plot(df_filtered["time_stamp"], df_filtered["health_score"], label="Vérité Terrain", color="blue")
plt.plot(df_filtered["time_stamp"], df_filtered["health_score_prediction"], label="Prédiction", color="red", linestyle="dashed")
plt.xticks(rotation=45)
plt.xlabel("Temps")
plt.ylabel("Health Score")
plt.legend()
plt.title(f"Évolution de Health Score pour ID {sensor_id}\n({start_date.date()} → {end_date.date()})")
plt.show()
