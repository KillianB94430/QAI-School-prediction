import pandas as pd


def calculate_health_score(row):
    """
    Calcule un score de santé environnemental basé sur les paramètres CO2, Humedad, PM2.5 et Temperatura.
    """
    # Scores individuels
    co2_score = max(0, 100 - (row['CO2'] - 400) / 6) if row['CO2'] <= 2000 else 0
    humedad_score = 100 if 30 <= row['Humedad'] <= 60 else max(0, 100 - abs(row['Humedad'] - 45) * 2)
    pm25_score = max(0, 100 - row['PM2.5'] * 5) if row['PM2.5'] <= 50 else 0
    temperatura_score = 100 if 20 <= row['Temperatura'] <= 25 else max(0, 100 - abs(row['Temperatura'] - 22.5) * 10)

    # Moyenne pondérée des scores
    health_score = (0.3* co2_score + 0.2 * humedad_score + 0.4 * pm25_score + 0.1 * temperatura_score)
    return round(health_score, 2)

def generate_features(input_path, output_path):
    """
    Charge les données brutes, calcule le score de santé environnemental et sauvegarde les données enrichies.
    """
    # Charger les données
    df = pd.read_csv(input_path)

    # Remplir les valeurs manquantes avec des moyennes 
    df['Humedad'] = df['Humedad'].fillna(df['Humedad'].mean())
    df['PM2.5'] = df['PM2.5'].fillna(df['PM2.5'].mean())
    df['Temperatura'] = df['Temperatura'].fillna(df['Temperatura'].mean())
    df['CO2'] = df['CO2'].fillna(df['CO2'].mean())

    # Sauvegarder les données après traitement des valeurs manquantes
    intermediate_csv = "../../data/processed/intermediate_data.csv"
    df.to_csv(intermediate_csv, index=False)
    print(f"Données intermédiaires sauvegardées dans {intermediate_csv}")

    # Calculer le score de santé
    df['health_score'] = df.apply(calculate_health_score, axis=1)

    # Sauvegarder les données enrichies
    df.to_csv(output_path, index=False)
    print(f"Données enrichies sauvegardées dans {output_path}")
    
if __name__ == "__main__":
    input_csv = "../../data/raw/final_merged_data.csv"
    output_csv = "../../data/processed/final_enriched_data.csv"

    print(f"Traitement des données depuis {input_csv}...")
    generate_features(input_csv, output_csv)
    print("Traitement terminé.")