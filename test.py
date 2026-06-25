import joblib

# ÉTAPE 1 : Charger le modèle et le vectoriseur
vectorizer = joblib.load(r'C:\Users\malek\spam mail\vectorizer.pkl')
model      = joblib.load(r'C:\Users\malek\spam mail\spam_model.pkl')

print("Modèle et vectoriseur chargés !")

# ÉTAPE 2 : Écrire le message à tester
# Change ce message par celui que tu veux tester
message = ["Free entry win prize call now!"]

# ÉTAPE 3 : Transformer le message en vecteur
X = vectorizer.transform(message)

print(f"\nMessage testé : {message[0]}")
print(f"Vecteur TF-IDF : {X.shape[1]} dimensions")

# ÉTAPE 4 : Prédire
prediction   = model.predict(X)
probabilites = model.predict_proba(X)

print(f"\nProbabilité Ham  : {probabilites[0][0] * 100:.2f}%")
print(f"Probabilité Spam : {probabilites[0][1] * 100:.2f}%")

# ÉTAPE 5 : Afficher le résultat
if prediction[0] == 1:
    print("\nRésultat : SPAM !")
else:
    print("\nRésultat : HAM (pas spam)")