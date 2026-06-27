# Spam Mail Classifier

Ce projet implémente un classificateur de SMS pour détecter le spam à partir du jeu de données `SMSSpamCollection.txt`.

## Structure du projet

- `SMSSpamCollection.txt` : dataset original avec les labels `spam` et `ham`
- `import.py` : lecture du dataset, conversion des labels, application de TF-IDF et sauvegarde des vecteurs
- `converter.py` : chargement des vecteurs, préparation des données et sauvegarde du jeu train/test
- `trainer.py` : entraînement du modèle Naive Bayes et évaluation sur les données de test
- `test.py` : test d'un message individuel avec le modèle entraîné
- `data_clean.csv` : dataset nettoyé sauvegardé par `import.py`
- `X_vectors.pkl` : vecteurs TF-IDF sauvegardés par `import.py`
- `data_split.pkl` : données split train/test sauvegardées par `converter.py`
- `vectorizer.pkl` : vectoriseur TF-IDF sauvegardé pour prédire de nouveaux messages
- `spam_model.pkl` : modèle Naive Bayes entraîné sauvegardé par `trainer.py`

## Prérequis

- Python 3.x
- Packages Python : `pandas`, `scikit-learn`, `joblib`

## Installation

Installez les dépendances avec pip :

```powershell
pip install pandas scikit-learn joblib
```

## Usage

1. Préparer les données et générer les vecteurs :

```powershell
python import.py
```

2. Diviser les données en train/test :

```powershell
python converter.py
```

3. Entraîner le modèle et évaluer :

```powershell
python trainer.py
```

4. Tester un message individuel :

```powershell
python test.py
```

## Personnalisation

- Modifiez le message dans `test.py` pour tester un autre SMS.
- Si vous déplacez le projet, mettez à jour les chemins absolus dans les scripts ou adaptez-les en chemins relatifs.

## Résultat attendu

- `spam_model.pkl` : modèle entraîné
- `vectorizer.pkl` : transformateur TF-IDF
- `data_clean.csv`, `X_vectors.pkl`, `data_split.pkl` : données préparées

## À propos

Ce projet est un exemple simple de pipeline de classification de spam. Il illustre la préparation de données, la vectorisation TF-IDF, la division train/test et l'entraînement d'un modèle Naive Bayes.
