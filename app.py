import streamlit as st
import pandas as pd
# ... tous vos imports qui fonctionnent ...

st.set_page_config(layout="wide")
st.title("Test avec Secrets")

try:
    FMP_API_KEY = st.secrets["FMP_API_KEY"]
    st.write(f"Clé API FMP lue : {FMP_API_KEY[:5]}...") # Affiche les 5 premiers caractères pour confirmation
except Exception as e:
    st.error(f"Erreur de lecture du secret FMP: {e}")

st.write("Fin du test de secrets.")
    
