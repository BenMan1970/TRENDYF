# ... (début du script, imports, config, initialisation td, fonctions, etc.)

st.set_page_config(layout="wide")
# ... (logs initiaux)

st.title("Classement des Paires Forex par Tendance MTF (via Twelve Data)")
st.write("Interface Streamlit initialisée (Titre OK).")

# Initialiser session_state si ce n'est pas déjà fait
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_results' not in st.session_state: # Aussi initialiser df_results pour éviter KeyError
    st.session_state.df_results = pd.DataFrame()


bouton_clique = st.button("Actualiser et Analyser les Paires Forex (Twelve Data)")
st.write(f"État du bouton avant le if: {bouton_clique}") # LOG POUR VOIR LA VALEUR DU BOUTON

if bouton_clique:
    st.write("Bouton cliqué - DANS LE IF.") # LOG CLIC BOUTON
    st.session_state.data_loaded = False # Reset flag
    st.session_state.df_results = pd.DataFrame() # Vider les anciens résultats

    st.write("Avant d'appeler analyze_forex_pairs_twelvedata...")
    # On enlève le spinner temporairement pour voir si les logs s'affichent directement
    # with st.spinner("Analyse des données Twelve Data en cours... Merci de patienter."):
    # st.write("Spinner activé, appel de analyze_forex_pairs_twelvedata...")
    
    df_results_temp = analyze_forex_pairs_twelvedata() # Appel direct
    
    st.write(f"analyze_forex_pairs_twelvedata a retourné : {type(df_results_temp)}")
    if df_results_temp is not None and not df_results_temp.empty:
        st.session_state.df_results = df_results_temp
        st.session_state.data_loaded = True
        st.write("Résultats stockés dans session_state.")
    else:
        st.warning("Aucun résultat retourné par l'analyse ou df vide.")
        st.session_state.data_loaded = False

# Affichage conditionnel basé sur st.session_state
st.write(f"État de data_loaded avant l'affichage: {st.session_state.data_loaded}")

if st.session_state.data_loaded:
    st.write("Affichage des résultats depuis session_state...")
    df_to_display = st.session_state.get('df_results')
    if df_to_display is not None and not df_to_display.empty:
        # ... (code d'affichage du dataframe stylé et résumé)
        st.subheader("Classement des paires Forex")
        def style_trends(val):
            if val == 'Bullish': return 'background-color: mediumseagreen; color: white;'
            elif val == 'Bearish': return 'background-color: indianred; color: white;'
            elif val == 'Neutral': return 'background-color: khaki; color: black;'
            else: return 'background-color: whitesmoke; color: black;'
        styled_df = df_to_display.style.applymap(style_trends, subset=['H1', 'H4', 'D', 'W'])
        st.dataframe(styled_df, use_container_width=True)

        st.subheader("Résumé des Indicateurs")
        st.markdown("""
        - **H1 (1 Heure)**: Tendance basée sur HMA(12) vs EMA(20).
        - **H4 (4 Heures)**: Tendance basée sur HMA(12) vs EMA(20).
        - **D (Journalier)**: Tendance basée sur EMA(20) vs EMA(50).
        - **W (Hebdomadaire)**: Tendance basée sur EMA(20) vs EMA(50).
        - **Score**: Somme des tendances (+1 pour Bullish, -1 pour Bearish, 0 pour Neutral) sur les 4 timeframes.
        """)
    else:
        st.warning("Les données étaient marquées comme chargées mais sont vides ou None.")

elif not bouton_clique: # Si le bouton n'a PAS encore été cliqué dans CETTE exécution du script
    st.info("Cliquez sur le bouton ci-dessus pour lancer l'analyse avec les données Twelve Data.")
# Si le bouton a été cliqué mais que data_loaded est False (analyse échouée ou retourné vide),
# le message st.warning("Aucun résultat retourné...") devrait déjà s'être affiché.

# Assurez-vous que les fonctions analyze_forex_pairs_twelvedata, get_twelvedata_data, hma, get_trend sont définies
# ... (votre code des fonctions ici, AVEC les st.write de débogage à l'intérieur) ...
