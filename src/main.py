import subprocess
import os

# Dossiers contenant les scripts
training_dir = "./training_pipeline/"
inference_dir = "./inference_pipeline/"

# Demande de la date à l'utilisateur une seule fois
start_date = input("📆 Entrez la date de début (YYYY-MM-DD) : ")
end_date = input("📆 Entrez la date de fin (YYYY-MM-DD) : ")

# Liste des scripts à exécuter dans l'ordre
scripts = [
    (training_dir, "feature_engineering.py", []),
    (training_dir, "model_training.py", []),
    (inference_dir, "inf_feature_engineering.py", []),
    (inference_dir, "model_inference.py", [start_date, end_date]),
    (inference_dir, "compare_predictions.py", [start_date, end_date]),
]

for script_dir, script, args in scripts:
    script_path = os.path.join(script_dir, script)
    print(f"\n🚀 Exécution de {script_path}...\n")
    try:
        # Définir le répertoire de travail pour le script
        working_dir = os.path.abspath(script_dir)
        
        # Exécution avec arguments si nécessaire
        result = subprocess.run(
            ["python", script] + args,  # Passe les arguments si disponibles
            cwd=working_dir,  # Définit le répertoire de travail
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)  # Affiche la sortie du script
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'exécution de {script_path} :\n{e.stderr}")
        break  # Arrêter si une erreur se produit
