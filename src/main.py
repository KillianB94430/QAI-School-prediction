import subprocess
import os

# Dossiers contenant les scripts
training_dir = "./training_pipeline/"
inference_dir = "./inference_pipeline/"

# Liste des scripts à exécuter dans l'ordre
scripts = [
    (training_dir, "feature_engineering.py"),
    (training_dir, "model_training.py"),
    (inference_dir, "inf_feature_engineering.py"),
    (inference_dir, "model_inference.py"),
    (inference_dir, "compare_predictions.py"),
]

for script_dir, script in scripts:
    script_path = os.path.join(script_dir, script)
    print(f"\n🚀 Exécution de {script_path}...\n")
    try:
        # Définir le répertoire de travail pour le script
        working_dir = os.path.abspath(script_dir)
        result = subprocess.run(
            ["python", script],
            cwd=working_dir,  # Définit le répertoire de travail
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)  # Affiche la sortie du script
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'exécution de {script_path} :\n{e.stderr}")
        break  # Arrêter si une erreur se produit