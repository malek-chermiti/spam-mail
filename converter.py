import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# ÉTAPE 1 : Charger X déjà transformé
X = joblib.load(r'C:\Users\malek\spam mail\X_vectors.pkl')

print(f"X chargé : {X.shape[0]} messages x {X.shape[1]} mots")

# ÉTAPE 2 : Charger les labels
df = pd.read_csv(r'C:\Users\malek\spam mail\data_clean.csv')
y = df['label']

print(f"Labels chargés : {len(y)} labels")
print(f"Spam  : {y.sum()} messages")
print(f"Ham   : {len(y) - y.sum()} messages")

# ÉTAPE 3 : Diviser en 80% train et 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nDonnées entraînement : {X_train.shape[0]} messages")
print(f"Données test         : {X_test.shape[0]} messages")

# ÉTAPE 4 : Sauvegarder
joblib.dump((X_train, X_test, y_train, y_test),
            r'C:\Users\malek\spam mail\data_split.pkl')

print("\ndata_split.pkl contient :")
print(f"├── X_train → {X_train.shape[0]} vecteurs pour entraîner")
print(f"├── X_test  → {X_test.shape[0]} vecteurs pour tester")
print(f"├── y_train → {len(y_train)} labels pour entraîner")
print(f"└── y_test  → {len(y_test)} labels pour tester")
print("\nconverter.py terminé !")