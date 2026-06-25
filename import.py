import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ÉTAPE 1 : Charger le dataset
df = pd.read_csv(r'C:\Users\malek\spam mail\SMSSpamCollection.txt', 
                 sep='\t', 
                 header=None, 
                 names=['label', 'message'])

print(f"Dataset chargé : {len(df)} messages")

# ÉTAPE 2 : Convertir les labels en chiffres
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

print("Labels convertis : spam=1, ham=0")

# ÉTAPE 3 : Appliquer TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])

print(f"TF-IDF appliqué : {X.shape[0]} messages x {X.shape[1]} mots")

# ÉTAPE 4 : Sauvegarder
joblib.dump(vectorizer, r'C:\Users\malek\spam mail\vectorizer.pkl')#sait transformer n'importe quel nouveau message
joblib.dump(X,          r'C:\Users\malek\spam mail\X_vectors.pkl')#contient déjà les 5572 messages transformés
df.to_csv(              r'C:\Users\malek\spam mail\data_clean.csv', index=False)

print("\nvectorizer.pkl → vocabulaire de 8713 mots sauvegardé")
print("X_vectors.pkl  → vecteurs [0, 0, 0.8, ...] sauvegardés")
print("data_clean.csv → messages + labels sauvegardés")
print("\nimport.py terminé !")