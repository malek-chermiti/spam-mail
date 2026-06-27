# Spam Mail Classifier

Ce projet impl�mente un classificateur de SMS pour d�tecter le spam � partir du jeu de donn�es `SMSSpamCollection.txt`.

## Structure du projet

- `SMSSpamCollection.txt` : dataset original avec les labels `spam` et `ham`
- `import.py` : lecture du dataset, conversion des labels, application de TF-IDF et sauvegarde des vecteurs
- `converter.py` : chargement des vecteurs, pr�paration des donn�es et sauvegarde du jeu train/test
- `trainer.py` : entra�nement du mod�le Naive Bayes et �valuation sur les donn�es de test
- `test.py` : test d'un message individuel avec le mod�le entra�n�
- `data_clean.csv` : dataset nettoy� sauvegard� par `import.py`
- `X_vectors.pkl` : vecteurs TF-IDF sauvegard�s par `import.py`
- `data_split.pkl` : donn�es split train/test sauvegard�es par `converter.py`
- `vectorizer.pkl` : vectoriseur TF-IDF sauvegard� pour pr�dire de nouveaux messages
- `spam_model.pkl` : mod�le Naive Bayes entra�n� sauvegard� par `trainer.py`

## Pr�requis

- Python 3.x
- Packages Python : `pandas`, `scikit-learn`, `joblib`

## Installation

Installez les d�pendances avec pip :

```powershell
pip install pandas scikit-learn joblib
```

## Usage

1. Pr�parer les donn�es et g�n�rer les vecteurs :

```powershell
python import.py
```

2. Diviser les donn�es en train/test :

```powershell
python converter.py
```

3. Entra�ner le mod�le et �valuer :

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
- Si vous d�placez le projet, mettez � jour les chemins absolus dans les scripts ou adaptez-les en chemins relatifs.

## R�sultat attendu

- `spam_model.pkl` : mod�le entra�n�
- `vectorizer.pkl` : transformateur TF-IDF
- `data_clean.csv`, `X_vectors.pkl`, `data_split.pkl` : donn�es pr�par�es

## � propos

Ce projet est un exemple simple de pipeline de classification de spam. Il illustre la pr�paration de donn�es, la vectorisation TF-IDF, la division train/test et l'entra�nement d'un mod�le Naive Bayes.
