from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ÉTAPE 1 : Charger les données
X_train, X_test, y_train, y_test = joblib.load(
    r'C:\Users\malek\spam mail\data_split.pkl'
)

print(f"Données chargées !")
print(f"X_train : {X_train.shape[0]} messages pour entraîner")
print(f"X_test  : {X_test.shape[0]} messages pour tester")

# ÉTAPE 2 : Créer le modèle
model = MultinomialNB()

print("\nModèle créé : Naive Bayes")
print("Le modèle va apprendre les probabilités...")

# ÉTAPE 3 : Entraîner le modèle
model.fit(X_train, y_train)

print("Modèle entraîné !")
# ÉTAPE 4 : Évaluer le modèle
y_pred = model.predict(X_test)

print(f"\nRésultats sur {X_test.shape[0]} messages de test :")
print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nRapport détaillé :")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# ÉTAPE 5 : Sauvegarder le modèle
joblib.dump(model, r'C:\Users\malek\spam mail\spam_model.pkl')

print("spam_model.pkl sauvegardé !")
print("\ntrainer.py terminé !")