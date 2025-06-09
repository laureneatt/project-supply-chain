import streamlit as st
from src.accueil import page_accueil
from src.presentation import page_presentation_donnees
from src.exploration import page_exploration
from src.modelisation import page_modelisation
from src.optimisation import page_optimisation
from src.conclusion import page_conclusion

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à", [
        "Introduction", 
        "Présentation des données", 
        "Exploration", 
        "Modélisation", 
        "Optimisation", 
        "Conclusion"
    ])

    if page == "Introduction":
        page_accueil()
    elif page == "Présentation des données":
        page_presentation_donnees()
    elif page == "Exploration":
        page_exploration()
    elif page == "Modélisation":
        page_modelisation()
    elif page == "Optimisation":
        page_optimisation()
    elif page == "Conclusion":
        page_conclusion()

if __name__ == "__main__":
    main()
