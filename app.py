# --- Interface Utilisateur Streamlit ---
# ... (le début est inchangé) ...

if st.session_state.analysis_done_once:
    if not st.session_state.df_results.empty:
        st.subheader("Classement des paires Forex")
        df_to_display = st.session_state.df_results.copy() # Utiliser .copy() est une bonne pratique
        
        # Pour le diagnostic, vous pouvez décommenter la ligne suivante pour voir le DataFrame brut
        # st.write("DataFrame avant le nettoyage et le style :")
        # st.write(df_to_display)

        def style_trends(val):
            if val == 'Bullish': return 'background-color: #2E7D32; color: white;'
            elif val == 'Bearish': return 'background-color: #C62828; color: white;'
            elif val == 'Neutral': return 'background-color: #FFD700; color: black;'
            else: return ''
        
        # --- DEBUT DE LA CORRECTION ---

        # ÉTAPE 1 (LA CORRECTION PRINCIPALE): Remplacer les valeurs manquantes (NaN)
        # dans la colonne 'Paire' par un texte explicite.
        df_to_display['Paire'] = df_to_display['Paire'].fillna('Paire Manquante')

        # ÉTAPE 2: Mettre à jour .applymap en .map pour corriger le FutureWarning de vos logs.
        # La syntaxe est la même pour ce cas d'usage.
        styled_df = df_to_display.style.map(style_trends, subset=['H1', 'H4', 'D', 'W'])

        # Le calcul de la hauteur dynamique reste utile pour tout afficher
        height_dynamic = (len(df_to_display) + 1) * 35 + 3

        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=height_dynamic)
        
        # --- FIN DE LA CORRECTION ---

        st.subheader("Résumé des Indicateurs")
        st.markdown("""
        - **H1, H4**: Tendance basée sur HMA(12) vs EMA(20).
        - **D, W**: Tendance basée sur EMA(20) vs EMA(50).
        """)
    else:
        st.info("L'analyse n'a produit aucun résultat. Vérifiez les notifications pour d'éventuelles erreurs.")
elif not st.session_state.analysis_done_once:
    st.info("Cliquez sur 'Analyser' pour charger les données et voir le classement.")

st.markdown("---")
st.caption("Données via FinancialModelingPrep API.")

