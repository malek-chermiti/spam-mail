import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ÉTAPE 1 : Charger le dataset
df = pd.read_csv(r'C:\Users\malek\spam mail\SMSSpamCollection.txt', 
                 sep='\t', 
                 header=None, 
                 names=['label', 'message'])

print("Dataset chargé !")
print(f"Nombre de messages : {len(df)}")
print(df.head())

# ÉTAPE 2 : Convertir les labels en chiffres
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

print("\nLabels convertis :")
print(df['label'].value_counts())

# ÉTAPE 3 : Appliquer TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])

print(f"\nVocabulaire appris : {X.shape[1]} mots")
print(f"Taille du vecteur : {X.shape}")

# ÉTAPE 4 : Sauvegarder
joblib.dump(vectorizer, r'C:\Users\malek\spam mail\vectorizer.pkl')
df.to_csv(r'C:\Users\malek\spam mail\data_clean.csv', index=False)

print("\nvectorizer.pkl sauvegardé !")
print("data_clean.csv sauvegardé !")
print("\nimport.py terminé !")