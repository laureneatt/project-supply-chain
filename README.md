# project-supply-chain
## Analyse et Prédiction de la Satisfaction Client à partir des Avis Consommateurs

## Table des matières  
1. [Contexte](#contexte)  
2. [Objectif du Projet](#objectif-du-projet)  
3. [Données](#données)  
4. [Étapes du Projet](#étapes-du-projet)  
5. [Ressources Supplémentaires](#ressources-supplémentaires)

---

## Contexte

Dans le cadre de la formation **DataScientest**, ce projet vise à appliquer des méthodes de traitement automatique du langage naturel (NLP) et de machine learning à des données textuelles issues de la supply chain.

La **satisfaction client** est un indicateur clé dans la gestion de la Supply Chain. Elle est influencée par des facteurs comme la livraison, la qualité des produits ou la réactivité du service client. Les **avis en ligne** représentent une source précieuse pour comprendre les attentes des consommateurs et identifier les points de friction. Cependant, leur nature non structurée (commentaires textuels, étoiles, réponses) rend leur exploitation complexe sans traitement automatisé.

---

## Objectif du Projet

- Préparer et nettoyer un **jeu de données d’avis clients**  
- Analyser les **commentaires textuels**
- Prédire la **note de satisfaction (1 à 5 étoiles)**
- Proposer des pistes pour optimiser le traitement automatique des avis clients

---

## Données

Les données proviennent de sources d'avis clients telles que **Trustpilot** et **Trusted Shops**, et sont issues de deux entreprises : **Veepee** et **ShowRoom**.

### Description des Variables

#### 🎯 Variable cible (`y`)
| Nom   | Description |
|-------|-------------|
| `star` | Note de satisfaction du client (de 1 à 5) |

#### 🔍 Variables explicatives (`X`)
| Nom              | Description |
|------------------|-------------|
| `Commentaire`    | Avis textuel du client |
| `réponse`        | Réponse de l’entreprise au commentaire |
| `company`        | Nom de l’entreprise (Veepee ou ShowRoom) |
| `source`         | Plateforme d’avis (Trustpilot) |
| `ville`          | Ville renseignée par le client |
| `ecart`          | Écart entre la date de commande et la date d’avis |
| `date_commande`  | Date de la commande |
| `date`           | Date de publication de l’avis |
| `maj`            | Date de mise à jour éventuelle de l’avis |
| `client`         | Identifiant du client (sera supprimé) |

---

## Étapes du Projet

### 📒 Notebook 1 : *Exploration et Préparation des Données Clients*

1. **Chargement et Exploration Initiale**
2. **Description et Analyse des Variables**
3. **Prétraitement des Données**
4. **Encodage des Variables Catégorielles**
5. **Traitement des Commentaires**
6. **Gestion des Valeurs Manquantes**
7. **Vérifications et Export**

---

### 🤖 Notebook 2 : *Modélisation des Notes Clients*

1. **Chargement et Exploration du Dataset**
2. **Préparation des données**
3. **Rééquilibrage des Classes**
4. **Modélisation**
5. **Évaluation**
6. **Interprétation et pistes d’amélioration**

---

### 📱 Création d’une application Streamlit

- Développement d’une interface pour visualiser les études effectuées :  
  - Résultats d’analyse exploratoire  
  - Visualisations des données et statistiques clés  
  - Synthèses des modèles de prédiction  
- Objectif : rendre les résultats et analyses accessibles et interactifs  
- Outils utilisés : `streamlit`, `pandas`, `matplotlib`/`seaborn`  .. (à ajouter)

  ## Ressources Supplémentaires

- [Trustpilot](https://www.trustpilot.com/)
- [spaCy NLP](https://spacy.io/)
- [SMOTE - Imbalanced-learn](https://imbalanced-learn.org/stable/over_sampling.html#smote)
