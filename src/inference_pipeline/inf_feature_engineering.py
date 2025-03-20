import pandas as pd


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


    # Sauvegarder les données enrichies
    df.to_csv(output_path, index=False)
    print(f"Données enrichies sauvegardées dans {output_path}")
    
if __name__ == "__main__":
    input_csv = "../../data/raw/final_merged_data.csv"
    output_csv = "../../data/processed/inference_feature_data.csv"

    print(f"Traitement des données depuis {input_csv}...")
    generate_features(input_csv, output_csv)
    print("Traitement terminé.")