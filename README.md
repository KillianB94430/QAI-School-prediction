# QAI-School-Prediction

## Description
Ce projet vise à analyser et prédire les niveaux de pollution intérieure (à partir de mesures de CO2, humidité, particules fines, température) en utilisant des techniques de Machine Learning et Deep Learning. Il intègre également **MLflow** pour le suivi des expériences et l'optimisation des modèles.

## Fonctionnalités
- Chargement et prétraitement des données
- Entraînement de plusieurs modèles (Random Forest, XGBoost, BiLSTM)
- Comparaison des modèles et sauvegarde du meilleur
- Suivi des performances avec MLflow
- Inférence sur de nouvelles données et génération de prédictions

## Installation
Assurez-vous d'avoir **Python 3.8+** installé, puis exécutez les commandes suivantes pour installer les dépendances :

```bash
# Cloner le projet
git clone https://github.com/KillianB94430/QAI-School-prediction.git
cd QAI-School-prediction

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation
### 1. **Entraînement du modèle**
Lancez l'entraînement et le suivi avec MLflow :
```bash
python src/model_training.py
```
Le meilleur modèle sera sauvegardé dans **MLflow**.

### 2. **Inférence sur de nouvelles données**
Une fois le modèle entraîné, vous pouvez l'utiliser pour effectuer des prédictions :
```bash
python src/inference_pipeline/model_inference.py
```
Les prédictions seront sauvegardées dans `data/processed/inference_predictions.csv`.

## Architecture du projet
```
QAI-School-prediction/
│── data/                   # Données brutes et traitées
│── src/                    # Scripts d'entraînement et d'inférence
│   │── training_pipeline/  # Entraînement et sauvegarde des modèles
│   │── inference_pipeline/ # Scripts d'inférence
    │── main.py             # Modèles sauvegardés
    │── mlruns/             # Expériences MLflow
│── requirements.txt        # Dépendances du projet
│── README.md               # Documentation
```

## Suivi des Expériences avec MLflow
Le projet utilise **MLflow** pour enregistrer les modèles et les métriques. Vous pouvez visualiser les résultats avec :
```bash
mlflow ui
```
Ensuite, ouvrez **http://127.0.0.1:5000** dans votre navigateur.



---
*Projet réalisé par Killian B. et Louis G.*

