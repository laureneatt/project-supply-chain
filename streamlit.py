# -*- coding: utf-8 -*-
"""
Created on Thu May 22 15:48:18 2025

@author: momatallah
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Séparation train/test
from sklearn.model_selection import train_test_split

# Vectorisation TF-IDF du texte
from sklearn.feature_extraction.text import TfidfVectorizer

# Évaluation des modèles (rapport, matrice de confusion, précision)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Rééquilibrage des classes avec SMOTE
from imblearn.over_sampling import SMOTE
from collections import Counter

# Modèles classiques de machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


# Stopwords français pour le prétraitement textuel
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

################ partie 1 ###############################☺ 

# Chargement du fichier CSV contenant les avis nettoyés
df = pd.read_csv('C:/Users/momatallah/Desktop/reviews_trust_cleaned_final.csv')

# Affichage d’un aperçu des premières lignes pour vérifier le contenu
display(df.head())


############## partie 2 ####################### 

# Vérification des valeurs manquantes dans toutes les colonnes
missing_values = df.isna().sum()
print(missing_values)

# Nombre de valeurs manquantes dans la colonne 'Commentaire_clean' avant traitement
missing_before = df['Commentaire_clean'].isna().sum()
print(f"Valeurs manquantes avant traitement : {missing_before}")

# Remplacement des valeurs manquantes dans 'Commentaire_clean' par des chaînes vides
df['Commentaire_clean'] = df['Commentaire_clean'].fillna('')

# Vérification après traitement
missing_after = df['Commentaire_clean'].isna().sum()
print(f"Valeurs manquantes après traitement : {missing_after}")


################## Analyse de la distribution des notes  ############### 

plt.figure(figsize=(6,4))
sns.countplot(x='star', data=df, hue='star', palette='viridis', legend=False)
plt.title("Répartition des Notes (étoiles)")
plt.xlabel("Note")
plt.ylabel("Nombre de commentaires")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Résumé chiffré et analyse rapide
total_avis = len(df)
counts = df['star'].value_counts().sort_index()
percentages = counts / total_avis * 100

print("Nombre d'avis par note :")
for note, count in counts.items():
    print(f"Note {note} : {count} avis ({percentages[note]:.1f}%)")

print("\nObservations :")
print("- Les notes 1 et 5 représentent la majorité des avis, tandis que les notes intermédiaires sont moins fréquentes.")
print("- Cette répartition crée un déséquilibre des classes.")
print("\nNous devrons donc explorer des techniques adaptées pour gérer ce déséquilibre lors de la modélisation.")

##################  Séparation du jeu en train et test avec stratification ############ 

X = df['Commentaire_clean']
y = df['star']

# Division du dataset en jeu d'entraînement et de test
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

############  Vectorisation TF-IDF avec suppression des stopwords ########### 

# Chargement des stopwords français pour ignorer les mots fréquents peu informatifs
stopwords_fr = stopwords.words('french')

# Initialisation du vectoriseur TF-IDF 
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # Utilisation des unigrams et bigrams pour mieux capter le contexte
    max_features=10000, # Limitation à 10 000 tokens pour optimiser le temps de calcul
    stop_words=stopwords_fr #  Suppression des stopwords pour épurer le texte
)

# Entraînement du vectoriseur sur le texte d'entraînement et transformation en vecteurs numériques
X_train_vect = vectorizer.fit_transform(X_train_text)

# Transformation du texte de test avec le même vocabulaire (mêmes dimensions)
X_test_vect = vectorizer.transform(X_test_text)

################### Application de SMOTE sur le jeu d’entraînement ################# 

# Le jeu d'entraînement est déséquilibré : certaines notes (notamment 2, 3 et 4 étoiles)
# sont sous-représentées, ce qui peut nuire à la qualité des prédictions.

# SMOTE (Synthetic Minority Over-sampling Technique) permet de générer des exemples 
# synthétiques pour les classes minoritaires afin d’équilibrer le jeu d’entraînement.




############# Application de ADASYN ################### 

from imblearn.over_sampling import ADASYN
import pandas as pd

# Application d'ADASYN sur le jeu d'entraînement vectorisé
adasyn = ADASYN(random_state=42)
X_train_res, y_train_res = adasyn.fit_resample(X_train_vect, y_train)

# Vérification de la distribution des classes avant et après rééquilibrage
print("Distribution avant ADASYN :")
print(pd.Series(y_train).value_counts().sort_index())

print("\nDistribution après ADASYN :")
print(pd.Series(y_train_res).value_counts().sort_index())

################# Création d’une fonction d’entraînement et d’évaluation ########## 

def test_model(clf, X_train, y_train, X_test, y_test):
    """
    Entraîne le modèle, prédit sur les données de test, affiche les métriques et la matrice de confusion,
    puis retourne un résumé des métriques principales.
    """
    clf.fit(X_train, y_train)   # Entraîne le modèle 'clf' sur les données d'entraînement
    y_pred = clf.predict(X_test) # Effectue les prédictions sur les données de test
    
    print(f"\n=== Résultats pour {clf.__class__.__name__} ===")    # Affiche le nom du modèle utilisé
    print(classification_report(y_test, y_pred))   # Affiche le rapport complet des métriques (précision, rappel, F1-score, etc.)
    
    # Affiche la matrice de confusion pour visualiser les erreurs de classification
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matrice de confusion - {clf.__class__.__name__}")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.show()
    
    # Calcule les métriques principales pour un résumé et une comparaison facile
    accuracy = accuracy_score(y_test, y_pred)   # Précision globale du modèle
    report = classification_report(y_test, y_pred, output_dict=True)  # Rapport des métriques sous forme de dictionnaire
    
    return {  # Retourne un résumé des métriques clés pour comparaison entre modèles
        'Model': clf.__class__.__name__,
        'Accuracy': accuracy,
        'Macro Precision': report['macro avg']['precision'],
        'Macro Recall': report['macro avg']['recall'],
        'Macro F1-score': report['macro avg']['f1-score']
        }

################# Entraînement, prédiction, et comparaison des performances ############### 

models = [
    LogisticRegression(max_iter=1000, random_state=42),
    MultinomialNB(),
    LinearSVC(random_state=42, max_iter=10000),
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    KNeighborsClassifier(n_neighbors=5)
]

# Liste pour stocker les résultats
results = []

# Entraînement et évaluation pour chaque modèle
for model in models:
    try:
        res = test_model(model, X_train_res, y_train_res, X_test_vect, y_test)
        results.append(res)
    except Exception as e:
        print(f"❌ Erreur avec {model.__class__.__name__} : {e}")

# Création d'un tableau comparatif trié par F1-score macro (moyenne sur toutes les classes)
if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Macro F1-score', ascending=False).reset_index(drop=True)

    print("\n✅ === Comparatif des modèles ===")
    print(df_results)
else:
    print("⚠️ Aucun résultat disponible. Tous les modèles ont échoué.")
