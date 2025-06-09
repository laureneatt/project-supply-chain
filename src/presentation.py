import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def page_presentation_donnees():
    st.title("📊 Présentation des Données")

    # 1. Contexte
    st.header("1. 📁 Contexte des données")
    st.markdown(
        """
        Les données analysées proviennent de deux plateformes d'avis clients reconnues : **Trustpilot** et **Trusted Shops**. 
        Elles concernent les retours d'expérience de clients de deux acteurs majeurs du e-commerce : **Veepee** et **ShowRoomPrivé**.

        Le fichier utilisé est : `reviews_trust.csv`.
        """
    )
    st.markdown("---")

    # 2. Chargement des données
    df = pd.read_csv('reviews_trust.csv', encoding='utf-8')

    st.header("2. 🔍 Aperçu et Qualité du Dataset")
    st.markdown(f"**Dimensions du dataset :** `{df.shape[0]:,}` lignes × `{df.shape[1]}` colonnes")

    st.subheader("📌 Aperçu des premières données")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("📋 Informations générales sur les colonnes")
    col_info = pd.DataFrame({
        "Colonne": df.columns,
        "Type": df.dtypes.astype(str),
        "Non-null": df.notnull().sum(),
        "Valeurs manquantes": df.isnull().sum(),
    })
    col_info["% Manquantes"] = (col_info["Valeurs manquantes"] / len(df) * 100).round(2)
    st.dataframe(col_info.reset_index(drop=True), use_container_width=True)

    st.subheader("🚨 Variables avec valeurs manquantes")
    missing_df = col_info[col_info["Valeurs manquantes"] > 0][["Colonne", "Valeurs manquantes", "% Manquantes"]]

    if not missing_df.empty:
        st.dataframe(missing_df.reset_index(drop=True), use_container_width=True)
    else:
        st.success("✅ Aucune valeur manquante détectée.")

    st.subheader("📈 Statistiques descriptives des colonnes numériques")
    numeric_desc = df.describe().transpose().round(2)
    st.dataframe(numeric_desc, use_container_width=True)

    st.markdown("---")

    # 3. Analyse initiale
    st.header("3. 🧠 Analyse initiale")
    st.markdown(
        f"""
        - **Nombre total d'avis :** `{df.shape[0]:,}`  
        - **Nombre de colonnes :** `{df.shape[1]}`  
        - **Note moyenne (`star`) :** `{df['star'].mean():.2f}` sur 5  
        - **Colonnes très incomplètes :** `maj`, `ville`, `date_commande`, `ecart`  
        - Ces variables devront être **supprimées ou traitées** selon leur utilité analytique.
        """
    )

    st.markdown("---")

    # 4. Distribution des notes
    st.header("4. ⭐ Distribution des Notes de Satisfaction")
    distribution_notes = df['star'].value_counts(normalize=True).sort_index() * 100

    for note, pct in distribution_notes.items():
        st.markdown(f"- **{int(note)} étoile{'s' if note > 1 else ''}** : `{pct:.2f}%` des avis")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x='star', data=df, palette='viridis', ax=ax)
    ax.set_title("Distribution des Notes de Satisfaction", fontsize=14)
    ax.set_xlabel("Note (étoiles)")
    ax.set_ylabel("Nombre d'avis")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)

    st.markdown(
        """
        **Observations :**  
        - La note **5 étoiles** est la plus fréquente (~40%), traduisant une forte satisfaction.  
        - À l'inverse, la note **1 étoile** représente près de **26%**, révélant une polarisation dans les avis.
        """
    )

    st.markdown("---")
    st.markdown(
        """
        ### 🔚 Conclusion préliminaire  
        Les données présentent une forte polarisation des avis, avec une majorité de clients très satisfaits, mais aussi un volume important d'avis très négatifs.  
        Les prochaines étapes viseront à approfondir l'analyse des causes possibles de cette polarisation, notamment via l'analyse textuelle des commentaires.
        """
    )


# Appel de la fonction (si ce fichier est exécuté directement)
if __name__ == "__main__":
    page_presentation_donnees()
