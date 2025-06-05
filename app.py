import time # Importer le module time en haut de votre script

# ... (début de votre script, imports, TDClient, fonctions hma, get_trend, get_twelvedata_data) ...

def analyze_forex_pairs_twelvedata():
    st.write("DEBUG analyze_forex_pairs_twelvedata: Début de l'analyse.")
    results = []
    timeframe_params = {
        'H1': {'interval': '1h',    'outputsize': 250},
        'H4': {'interval': '4h',    'outputsize': 250},
        'D':  {'interval': '1day',  'outputsize': 200},
        'W':  {'interval': '1week', 'outputsize': 100}
    }
    
    # Limite API Twelve Data (plan gratuit typique)
    API_CALLS_PER_MINUTE_LIMIT = 8 
    CALLS_PER_PAIR = len(timeframe_params) # 4 appels par paire
    
    api_calls_this_minute = 0

    for i, pair_symbol in enumerate(forex_pairs_twelvedata):
        st.write(f"DEBUG analyze_forex_pairs_twelvedata: Traitement de {pair_symbol} ({i+1}/{len(forex_pairs_twelvedata)}).")
        
        # Vérifier si on va dépasser la limite avec cette paire
        if api_calls_this_minute + CALLS_PER_PAIR > API_CALLS_PER_MINUTE_LIMIT:
            wait_time = 65 # Attendre un peu plus d'une minute pour être sûr
            st.write(f"DEBUG analyze_forex_pairs_twelvedata: Limite API atteinte. Pause de {wait_time} secondes...")
            time.sleep(wait_time)
            api_calls_this_minute = 0 # Réinitialiser le compteur d'appels pour la nouvelle minute
            st.write("DEBUG analyze_forex_pairs_twelvedata: Reprise après la pause.")


        try:
            data_sets = {}
            all_data_fetched_successfully = True
            for tf_key, params in timeframe_params.items():
                st.write(f"DEBUG analyze_forex_pairs_twelvedata: Récupération {tf_key} pour {pair_symbol}.")
                df = get_twelvedata_data(
                    symbol=pair_symbol,
                    interval=params['interval'],
                    output_size=params['outputsize']
                )
                api_calls_this_minute += 1 # Compter l'appel API

                if df.empty or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    st.error(f"analyze_forex_pairs_twelvedata: Données invalides ou OHLC manquantes pour {pair_symbol} ({tf_key}).")
                    all_data_fetched_successfully = False
                    break 
                data_sets[tf_key] = df
            
            if not all_data_fetched_successfully:
                st.write(f"DEBUG analyze_forex_pairs_twelvedata: Saut de {pair_symbol} car les données n'ont pas pu être toutes récupérées correctement.")
                # Si une récupération échoue pour une paire, on ne veut pas forcément réinitialiser api_calls_this_minute
                # car les appels précédents pour CETTE paire ont quand même été faits.
                # On pourrait décider de mettre une pause ici aussi si on veut être conservateur.
                continue

            # ... (le reste de votre logique de calcul d'indicateurs et de score) ...
            data_1h, data_4h, data_d, data_w = data_sets['H1'], data_sets['H4'], data_sets['D'], data_sets['W']
            st.write(f"DEBUG analyze_forex_pairs_twelvedata: Calcul des indicateurs pour {pair_symbol}.")

            if 'Close' not in data_1h or 'Close' not in data_4h or \
               'Close' not in data_d or 'Close' not in data_w:
                st.error(f"analyze_forex_pairs_twelvedata: Colonne 'Close' manquante pour {pair_symbol} avant calcul indicateurs.")
                continue

            hma12_1h = hma(data_1h['Close'], 12)
            ema20_1h = data_1h['Close'].ewm(span=20, adjust=False).mean()
            hma12_4h = hma(data_4h['Close'], 12)
            ema20_4h = data_4h['Close'].ewm(span=20, adjust=False).mean()
            ema20_d = data_d['Close'].ewm(span=20, adjust=False).mean()
            ema50_d = data_d['Close'].ewm(span=50, adjust=False).mean()
            ema20_w = data_w['Close'].ewm(span=20, adjust=False).mean()
            ema50_w = data_w['Close'].ewm(span=50, adjust=False).mean()

            indicators = [hma12_1h, ema20_1h, hma12_4h, ema20_4h, ema20_d, ema50_d, ema20_w, ema50_w]
            if any(s is None or s.dropna().empty for s in indicators):
                st.error(f"analyze_forex_pairs_twelvedata: Un ou plusieurs indicateurs sont vides/NaN pour {pair_symbol}.")
                continue

            trend_1h = get_trend(hma12_1h, ema20_1h)
            trend_4h = get_trend(hma12_4h, ema20_4h)
            trend_d = get_trend(ema20_d, ema50_d)
            trend_w = get_trend(ema20_w, ema50_w)

            score = sum([1 if t == 'Bullish' else -1 if t == 'Bearish' else 0 for t in [trend_1h, trend_4h, trend_d, trend_w]])
            st.write(f"DEBUG analyze_forex_pairs_twelvedata: Score {score} pour {pair_symbol}.")
            results.append({'Pair': pair_symbol, 'H1': trend_1h, 'H4': trend_4h, 'D': trend_d, 'W': trend_w, 'Score': score})

        except Exception as e:
            st.error(f"analyze_forex_pairs_twelvedata: Erreur générale DANS LA BOUCLE pour {pair_symbol}: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            # Si une erreur inattendue se produit pour une paire, on réinitialise le compteur pour la suivante
            # pour ne pas bloquer indéfiniment à cause d'une erreur de comptage.
            # Cependant, l'erreur TwelveDataError est déjà gérée par le fait que df sera vide.
            # Si c'est une autre erreur, on peut choisir de continuer ou d'arrêter.
            # Pour l'instant, on continue avec la paire suivante.
            api_calls_this_minute = 0 # Prudence pour la paire suivante en cas d'erreur non liée à l'API.
            continue
            
    st.write(f"DEBUG analyze_forex_pairs_twelvedata: Fin de l'analyse, {len(results)} résultats trouvés.")
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    return df

# ... (le reste de votre script Streamlit pour l'affichage) ...
