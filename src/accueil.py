import streamlit as st

def page_accueil():
    st.title("🚀 Supply Chain - Satisfaction des Clients")
    st.markdown("""
### 📊 Analyse et Prédiction de la Satisfaction Client à partir des Avis Consommateurs

**Contexte**  
Ce projet, réalisé dans le cadre de la formation **Datascientest**, a pour objectif d'explorer et d'analyser un jeu de données d'avis clients afin de mieux comprendre les facteurs influençant la satisfaction.  
En exploitant des données temporelles, textuelles et catégorielles, nous visons à extraire des insights actionnables pour améliorer l'expérience client.
""")

    st.markdown("---")

    st.header("🎯 Objectifs du Projet")
    st.markdown("""
- Préparer et nettoyer un jeu de données d’avis clients  
- Analyser les commentaires textuels  
- Prédire la note de satisfaction (1 à 5 étoiles)  
- Proposer des pistes pour optimiser le traitement automatique des avis clients  
""")

    st.markdown("---")

    st.header("📂 Données")
    st.markdown("""
Les données utilisées dans ce projet proviennent de sources d'avis clients telles que Trustpilot et Trusted Shops, et concernent deux entreprises : **Veepee** et **ShowRoom**, reconnues comme des acteurs majeurs dans le secteur de la vente en ligne et des ventes privées.

Fichier principal : `reviews_trust.csv`
""")

    st.subheader("🎯 Variable cible (y)")
    st.table({
        "Nom": ["star"],
        "Description": ["Note de satisfaction du client (de 1 à 5)"]
    })

    st.subheader("🔍 Variables explicatives (X)")
    st.table({
        "Nom": [
            "Commentaire", "réponse", "company", "source", "ville",
            "ecart", "date_commande", "date", "maj", "client"
        ],
        "Description": [
            "Avis textuel du client",
            "Réponse de l’entreprise au commentaire",
            "Nom de l’entreprise (Veepee ou ShowRoom)",
            "Plateforme d’avis (Trustpilot)",
            "Ville renseignée par le client",
            "Écart entre la date de commande et la date d’avis",
            "Date de la commande",
            "Date de publication de l’avis",
            "Date de mise à jour éventuelle de l’avis",
            "Identifiant du client (sera supprimé)"
        ]
    })

    st.markdown("---")

    st.header("🗂 Étapes du Projet")
    st.markdown("""
- Analyse exploratoire et nettoyage des données  
- Construction et entraînement des modèles de prédiction  
- Amélioration et optimisation des performances  
""")

    st.markdown("---")

    st.header("🔗 Ressources Supplémentaires")
    st.markdown("""
- [Trustpilot](https://fr.trustpilot.com/)  
- [spaCy NLP](https://spacy.io/)  
- [SMOTE - Imbalanced-learn](https://imbalanced-learn.org/)  
""")
