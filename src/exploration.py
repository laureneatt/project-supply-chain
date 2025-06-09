import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --- Fonctions utilitaires ---

def preprocess_dates(df):
    for col in ['date', 'maj', 'date_commande']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            prefix = 'commande' if col == 'date_commande' else col
            df[f'{prefix}_month'] = df[col].dt.month
            df[f'{prefix}_weekday'] = df[col].dt.weekday
            df[f'{prefix}_weekend'] = np.where(df[f'{prefix}_weekday'].isin([5, 6]), 1, 0)
            df[f'{prefix}_day'] = df[col].dt.day
            df[f'{prefix}_hour'] = df[col].dt.hour
    return df

def correlation_analysis(df):
    cols_corr = [
        'star',
        'date_month', 'date_weekday', 'date_weekend', 'date_day', 'date_hour',
        'commande_month', 'commande_weekday', 'commande_weekend', 'commande_day', 'commande_hour',
        'maj_month', 'maj_weekday', 'maj_weekend', 'maj_day', 'maj_hour'
    ]
    cols_present = [col for col in cols_corr if col in df.columns]
    if len(cols_present) < 2:
        return pd.DataFrame()  # Pas assez de colonnes pour corrélation
    return df[cols_present].corr()

def encode_categoricals(df):
    if 'company' in df.columns:
        df = pd.get_dummies(df, columns=['company'], prefix='company')
    if 'source' in df.columns:
        df = pd.get_dummies(df, columns=['source'], prefix='source')
    if 'ville' in df.columns:
        freq = df['ville'].value_counts(normalize=True)
        df['ville_encoded'] = df['ville'].map(freq)
    return df

def clean_text(text):
    custom_stopwords = set(ENGLISH_STOP_WORDS).union({
        "d", "même", "cest", "de", "je", "ne", "la", "le", "les", "des", "et", "à", "du", "un", "une", "c'est", "pour", "sur", "avec",
        "vous", "moi", "que", "comme", "mais", "a", "en", "dans", "si", "l", "il", "elle", "cela", "ça", "ce",
        "suis", "été", "ont", "au", "aux", "plus", "est", "sont", "nous", "mon", "me", "ses", "tout", "aujourd'hui",
        "pas", "qui", "jai", "aussi", "beaucoup", "donc", "encore", "site", "commande", "colis",
        "service", "livraison", "j", "ai", "ma", "par", "n", "ils", "enfin", "tres", "c", "qu", "m", "là", "peu", "t", "faire", "avis", "votre", "merci"
    })
    apostrophes = ["’", "‘", "´", "ʻ", "ʹ", "ʾ", "ʿ", "ˈ", "'"]

    if isinstance(text, str):
        text = text.lower()
        for apos in apostrophes:
            text = text.replace(apos, " ")
        text = ''.join(char for char in text if char not in string.punctuation and not char.isdigit())
        return ' '.join([w for w in text.split() if w not in custom_stopwords])
    return ''

# --- Fonction principale Streamlit ---

def page_exploration():
    st.title("📊 Analyse & Prétraitement des Données de Satisfaction Client")

    uploaded_file = st.file_uploader("1️⃣ Importez votre fichier CSV (avec colonnes attendues)", type=['csv'])

    if uploaded_file is not None:
        if 'df' not in st.session_state:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
        else:
            df = st.session_state.df

        st.write("📋 Aperçu des données brutes :")
        st.dataframe(df.head())

        # --- 2️⃣ Prétraitement des variables temporelles ---
        with st.expander("2️⃣ Prétraitement des variables temporelles"):
            df = preprocess_dates(df)
            st.session_state.df = df
            st.success("✅ Extraction des variables temporelles réalisée.")

            corr_matrix = correlation_analysis(df)
            if not corr_matrix.empty:
                st.write("Matrice de corrélation (variables temporelles vs 'star') :")
                st.dataframe(corr_matrix.style.format("{:.2f}"))

                fig, ax = plt.subplots(figsize=(10, 7))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            else:
                st.info("Pas assez de données temporelles pour calculer la corrélation.")

            st.write("Analyse des valeurs manquantes dans colonnes horaires :")
            cols_hours = [col for col in ['date_hour', 'commande_hour', 'maj_hour'] if col in df.columns]
            if cols_hours:
                st.write(df[cols_hours].info())
                st.write(df[cols_hours].head())
            else:
                st.write("Pas de colonnes horaires présentes.")

            st.markdown("""
            ### 🔎 Analyse des variables temporelles

            - Plusieurs variables liées à **'date'** et **'maj'** contiennent beaucoup de valeurs manquantes.
            - La majorité des variables présentent une corrélation très faible avec la variable cible **star**.
            - Seule la variable **commande_weekend** montre une corrélation positive modérée.
            - Les variables horaires (`date_hour`, `commande_hour`, `maj_hour`) sont trop incomplètes pour être exploitées efficacement.

            ---
            ✅ Nous conservons uniquement `commande_weekend` pour sa pertinence et supprimons les autres variables temporelles trop incomplètes ou peu corrélées.
            """)

            cols_to_drop = [
                'date', 'maj', 'date_month', 'date_weekday', 'date_weekend', 'date_day', 'date_hour',
                'maj_month', 'maj_weekday', 'maj_weekend', 'maj_day', 'maj_hour',
                'commande_month', 'commande_weekday', 'commande_day', 'commande_hour',
                'date_commande', 'ecart', 'month', 'weekday', 'day', 'hour'
            ]
            cols_present = [col for col in cols_to_drop if col in df.columns]
            df.drop(columns=cols_present, inplace=True)
            st.success("✅ Colonnes inutiles supprimées.")
            st.write(f"🗑️ Colonnes supprimées : {cols_present}")
            st.write(f"✅ Colonnes restantes : {df.columns.tolist()}")
            st.session_state.df = df

        # --- 3️⃣ Suppression de la variable 'client' ---
        with st.expander("3️⃣ Suppression de la variable 'client'"):
            df = st.session_state.df
            if 'client' in df.columns:
                st.markdown("""
                ### 6️⃣ Suppression de la variable `client`
                La variable `client` contient souvent des identifiants uniques qui n’apportent pas de valeur à l’analyse globale.  
                Nous la supprimons pour simplifier le jeu de données.
                """)
                st.write("📄 Colonnes avant suppression :", df.columns.tolist())
                df.drop('client', axis=1, inplace=True)
                st.success("✅ Variable `client` supprimée.")
                st.write("📄 Colonnes après suppression :", df.columns.tolist())
                st.session_state.df = df
            else:
                st.info("La colonne 'client' n'est pas présente.")

        # --- 4️⃣ Analyse de la variable cible 'star' ---
        with st.expander("4️⃣ Analyse de la variable cible 'star'"):
            df = st.session_state.df
            if 'star' in df.columns:
                star_dist = df['star'].value_counts(normalize=True).sort_index()
                st.write("📊 Répartition des notes (en %) :")
                st.bar_chart(star_dist)

                if 'company' in df.columns:
                    star_by_company = df.groupby('company')['star'].value_counts(normalize=True).unstack().fillna(0)
                    st.write("📈 Répartition des notes par entreprise :")
                    st.dataframe(star_by_company)

                    fig, ax = plt.subplots(figsize=(12,6))
                    star_by_company.plot(kind='bar', stacked=True, colormap='viridis', ax=ax)
                    ax.set_title("Répartition des Notes par Entreprise")
                    ax.set_xlabel("Entreprise")
                    ax.set_ylabel("Proportion")
                    st.pyplot(fig)

                    st.markdown("""
                    ### Interprétation des résultats :

                    - Environ 40% des clients sont très satisfaits (note 5), mais près de 25% sont insatisfaits (note 1).

                    - **ShowRoom** a une répartition équilibrée avec beaucoup de notes 5 et peu de notes 1, indiquant une bonne expérience client.

                    - **VeePee** montre surtout des notes 1, signalant un fort mécontentement.

                    - ShowRoom bénéficie donc d'une meilleure réputation, tandis que VeePee doit améliorer son service pour réduire l'insatisfaction.

                    """)
            else:
                st.info("La colonne 'star' est absente.")

        # --- 5️⃣ Encodage des variables catégorielles ---
        with st.expander("5️⃣ Encodage des variables catégorielles"):
            df = st.session_state.df
            df = encode_categoricals(df)
            st.session_state.df = df
            st.success("✅ Encodage One-Hot effectué pour 'company' et 'source'.")

            df_onehot = df.filter(regex='^(company_|source_)').astype(int)
            if not df_onehot.empty:
                st.write("### Aperçu des variables encodées")
                st.dataframe(df_onehot.head())

                st.write("### Répartition des catégories 'company' et 'source'")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("#### Company")
                    company_sum = df_onehot.filter(regex='^company_').sum().sort_values(ascending=False)
                    st.bar_chart(company_sum)

                with col2:
                    st.write("#### Source")
                    source_sum = df_onehot.filter(regex='^source_').sum().sort_values(ascending=False)
                    st.bar_chart(source_sum)
            else:
                st.info("Pas de colonnes encodées 'company' ou 'source' trouvées.")

            if 'ville' in df.columns:
                freq = df['ville'].value_counts(normalize=True)
                df['ville_encoded'] = df['ville'].map(freq)
                st.info("✅ Encodage par fréquence réalisé pour 'ville'.")

            if 'ville_encoded' in df.columns and 'star' in df.columns:
                corr_with_star = df[['star', 'ville_encoded']].corr().iloc[0, 1]
                corr_color = "green" if abs(corr_with_star) >= 0.05 else "red"
                st.markdown(f"📈 **Corrélation entre 'ville_encoded' et 'star' :** "
                            f"<span style='color:{corr_color}; font-weight:bold'>{corr_with_star:.3f}</span>",
                            unsafe_allow_html=True)

                if abs(corr_with_star) < 0.05:
                    df.drop(columns=['ville', 'ville_encoded'], inplace=True)
                    st.warning("🗑️ Variable 'ville' supprimée pour faible corrélation et données incomplètes.")

            st.write("### Aperçu du dataframe après encodage")
            st.dataframe(df.head())

            st.write("### Résumé des colonnes et types")
            info_df = pd.DataFrame({
                'Nom colonne': df.columns,
                'Type': df.dtypes.astype(str),
                'Valeurs non-null': df.notnull().sum().values
            })
            st.dataframe(info_df)

        # --- 6️⃣ Analyse et nettoyage des valeurs manquantes ---
        with st.expander("6️⃣ Analyse et nettoyage des valeurs manquantes"):
            df = st.session_state.df

            st.write("### Vérification initiale des valeurs manquantes :")
            missing_counts = df.isnull().sum()
            missing_percents = df.isnull().mean() * 100
            missing_summary = pd.DataFrame({
                'Nombre de valeurs manquantes': missing_counts,
                'Pourcentage (%)': missing_percents
            }).sort_values(by='Pourcentage (%)', ascending=False)
            st.dataframe(missing_summary)

            st.write("### Suppression des lignes sans commentaire et de la colonne 'reponse' (si présente)")
            before_rows = df.shape[0]
            df = df[df['Commentaire'].notna()].copy()
            after_rows = df.shape[0]
            st.write(f"Lignes supprimées (commentaires manquants): {before_rows - after_rows}")

            if 'reponse' in df.columns:
                df.drop(columns=['reponse'], inplace=True)
                st.write("Colonne 'reponse' supprimée.")

            st.write("### Vérification des valeurs manquantes après nettoyage :")
            missing_counts_after = df.isnull().sum()
            st.dataframe(missing_counts_after)

            st.session_state.df = df

        # --- 7️⃣ Vérification finale et suppression des doublons ---
        with st.expander("7️⃣ Vérification finale et suppression des doublons"):
            df = st.session_state.df
            st.write("### Types de données :")
            st.dataframe(df.dtypes)

            st.write("### Valeurs manquantes restantes :")
            st.dataframe(df.isnull().sum())

            before_dups = df.shape[0]
            df.drop_duplicates(inplace=True)
            after_dups = df.shape[0]
            st.write(f"Lignes supprimées suite à suppression des doublons : {before_dups - after_dups}")

            st.write(f"### Colonnes finales : {df.columns.tolist()}")
            st.write(f"### Forme finale du dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes")

            st.session_state.df = df

        # --- 8️⃣ Sauvegarde du dataset nettoyé final ---
        with st.expander("8️⃣ Sauvegarde du dataset nettoyé final"):
            df = st.session_state.df
            csv_final = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger le dataset nettoyé final (reviews_trust_cleaned_final.csv)",
                data=csv_final,
                file_name="reviews_trust_cleaned_final.csv",
                mime="text/csv"
            )
            st.success("Le dataset nettoyé a été sauvegardé avec succès.")

    else:
        st.info("⬆️ Veuillez importer un fichier CSV pour commencer l'analyse.")

if __name__ == "__main__":
    page_exploration()
