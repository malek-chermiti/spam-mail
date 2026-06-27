# Spam Mail Classifier

Ce projet implemente un classificateur de SMS pour detecter le spam e partir du jeu de donnees `SMSSpamCollection.txt`.

## Structure du projet

- `SMSSpamCollection.txt` : dataset original avec les labels `spam` et `ham`
- `import.py` : lecture du dataset, conversion des labels, application de TF-IDF et sauvegarde des vecteurs
- `converter.py` : chargement des vecteurs, preparation des donnees et sauvegarde du jeu train/test
- `trainer.py` : entraenement du modele Naive Bayes et evaluation sur les donnees de test
- `test.py` : test d'un message individuel avec le modele entraene
- `data_clean.csv` : dataset nettoye sauvegarde par `import.py`
- `X_vectors.pkl` : vecteurs TF-IDF sauvegardes par `import.py`
- `data_split.pkl` : donnees split train/test sauvegardees par `converter.py`
- `vectorizer.pkl` : vectoriseur TF-IDF sauvegarde pour predire de nouveaux messages
- `spam_model.pkl` : modele Naive Bayes entraene sauvegarde par `trainer.py`

## Prerequis

- Python 3.x
- Packages Python : `pandas`, `scikit-learn`, `joblib`

## Installation

Installez les dependances avec pip :

```powershell
pip install pandas scikit-learn joblib
```

## Usage

1. Preparer les donnees et generer les vecteurs :

```powershell
python import.py
```

2. Diviser les donnees en train/test :

```powershell
python converter.py
```

3. Entraener le modele et evaluer :

```powershell
python trainer.py
```

4. Tester un message individuel :

```powershell
python test.py
```

5. Lancer l'interface graphique :

```powershell
python interface.py
```

## Personnalisation

- Modifiez le message dans `test.py` pour tester un autre SMS.
- Si vous deplacez le projet, mettez e jour les chemins absolus dans les scripts ou adaptez-les en chemins relatifs.

## Resultat attendu

- `spam_model.pkl` : modele entraene
- `vectorizer.pkl` : transformateur TF-IDF
- `data_clean.csv`, `X_vectors.pkl`, `data_split.pkl` : donnees preparees

## e propos

Ce projet est un exemple simple de pipeline de classification de spam. Il illustre la preparation de donnees, la vectorisation TF-IDF, la division train/test et l'entraenement d'un modele Naive Bayes.
