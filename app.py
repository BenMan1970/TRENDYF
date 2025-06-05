# ... (imports et début du script comme avant) ...

# Fonction pour récupérer les données FMP (MODIFIÉE)
def get_fmp_data(symbol, interval_fmp, from_date_str=None, to_date_str=None, limit=None):
    pair_str_url = symbol
    st.write(f"DEBUG get_fmp_data: Appel pour {pair_str_url}, interval={interval_fmp}")
    
    base_url = "https://financialmodelingprep.com/api/v3/"
    params = {'apikey': FMP_API_KEY}
    data_key_in_json = None # Pour différencier la structure de réponse JSON

    if interval_fmp in ['1hour', '4hour']:
        endpoint = f"historical-chart/{interval_fmp}/{pair_str_url}"
        # Pas de from/to pour intraday ici, FMP retourne un historique récent
    elif interval_fmp == '1day': # Pour Daily
        # Utiliser l'endpoint historical-price-full pour daily Forex
        endpoint = f"historical-price-full/{pair_str_url}"
        data_key_in_json = "historical" # Les données journalières sont sous la clé "historical"
        if from_date_str: params['from'] = from_date_str
        if to_date_str: params['to'] = to_date_str
        if limit: params['serietype'] = 'line' # historical-price-full supporte from/to, et limit via serietype pour actions.
                                              # Pour Forex, on va se fier à from/to.
    else:
        st.error(f"Intervalle FMP non géré directement par get_fmp_data: {interval_fmp}. Weekly est géré par resample.")
        return pd.DataFrame()

    full_url = base_url + endpoint
    
    try:
        response = requests.get(full_url, params=params)
        response.raise_for_status()
        raw_data = response.json()
        st.write(f"DEBUG get_fmp_data: Appel API FMP à {full_url} effectué. Statut: {response.status_code}")

        if not raw_data or (isinstance(raw_data, dict) and raw_data.get('Error Message')):
            error_msg = raw_data.get('Error Message', 'Réponse vide ou invalide de FMP.') if isinstance(raw_data, dict) else 'Réponse vide de FMP.'
            st.warning(f"Aucune donnée FMP (ou erreur) pour {pair_str_url} ({interval_fmp}): {error_msg}")
            return pd.DataFrame()

        # Adapter la récupération des données en fonction de la structure JSON
        if data_key_in_json: # Pour historical-price-full (daily)
            if data_key_in_json in raw_data and isinstance(raw_data[data_key_in_json], list):
                data_list = raw_data[data_key_in_json]
            else:
                st.warning(f"Clé '{data_key_in_json}' non trouvée ou format incorrect dans la réponse FMP pour {pair_str_url} (daily).")
                return pd.DataFrame()
        else: # Pour historical-chart (intraday)
            data_list = raw_data

        if not data_list:
            st.warning(f"Liste de données vide après extraction JSON pour {pair_str_url} ({interval_fmp})")
            return pd.DataFrame()
            
        df = pd.DataFrame(data_list)
        
        if df.empty:
            st.warning(f"DataFrame vide après conversion pour {pair_str_url} ({interval_fmp})")
            return pd.DataFrame()

        if 'date' not in df.columns:
            st.error(f"Colonne 'date' manquante dans les données FMP pour {pair_str_url} ({interval_fmp})")
            return pd.DataFrame()
            
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        df.rename(columns=rename_map, inplace=True)
        df.sort_index(ascending=True, inplace=True)

        final_cols = ['Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns: final_cols.append('Volume')
        
        existing_final_cols = [c for c in final_cols if c in df.columns]
        
        if not all(c in existing_final_cols for c in ['Open', 'High', 'Low', 'Close']):
            st.error(f"Colonnes OHLC manquantes après traitement FMP pour {pair_str_url} ({interval_fmp}).")
            return pd.DataFrame()
            
        st.write(f"DEBUG get_fmp_data: Retour pour {pair_str_url} ({interval_fmp}) avec {len(df)} lignes.")
        return df[existing_final_cols]

    except requests.exceptions.HTTPError as http_err:
        st.warning(f"Erreur HTTP FMP pour {pair_str_url} ({interval_fmp}): {http_err}. Réponse: {response.text}")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Erreur DANS get_fmp_data pour {pair_str_url} ({interval_fmp}): {e}")
        import traceback
        st.warning(traceback.format_exc())
        return pd.DataFrame()

# MODIFICATION dans analyze_forex_pairs_fmp pour la gestion des dates pour Daily/Weekly
def analyze_forex_pairs_fmp():
    st.write("DEBUG analyze_fmp: Début de l'analyse.")
    results = []
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')

    timeframe_params_fmp = {
        'H1': {'interval_api': '1hour', 'days_history_for_daily_weekly': None}, 
        'H4': {'interval_api': '4hour', 'days_history_for_daily_weekly': None},
        'D':  {'interval_api': '1day',  'days_history_for_daily_weekly': 365 * 1}, # 1 an de données journalières
        'W':  {'interval_api': '1day',  'days_history_for_daily_weekly': 365 * 3}  # 3 ans de données journalières pour resampler en W
    }

    for pair_symbol in forex_pairs_fmp:
        st.write(f"DEBUG analyze_fmp: Traitement de {pair_symbol}.")
        try:
            data_sets = {}
            all_data_ok = True
            for tf_key, params in timeframe_params_fmp.items():
                from_date_req_str = None
                if params['days_history_for_daily_weekly']:
                    from_date_req = today - timedelta(days=params['days_history_for_daily_weekly'])
                    from_date_req_str = from_date_req.strftime('%Y-%m-%d')
                
                # L'intervalle à passer à l'API est toujours params['interval_api']
                # Le rééchantillonnage weekly se fera après avoir récupéré les données '1day'
                st.write(f"DEBUG analyze_fmp: Récupération {tf_key} (interval API FMP: {params['interval_api']}) pour {pair_symbol}.")
                df = get_fmp_data(
                    symbol=pair_symbol,
                    interval_fmp=params['interval_api'], # Utilise '1hour', '4hour', ou '1day'
                    from_date_str=from_date_req_str,
                    to_date_str=today_str if from_date_req_str else None
                )
                
                # Si tf_key est 'W', on a récupéré '1day' data, il faut resampler
                if tf_key == 'W' and not df.empty:
                    agg_funcs = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
                    if 'Volume' in df.columns: agg_funcs['Volume'] = 'sum'
                    df_weekly = df.resample('W-FRI').agg(agg_funcs) # W-FRI: Semaine se terminant Vendredi
                    df = df_weekly.dropna() # Enlever les semaines sans données complètes
                    st.write(f"DEBUG analyze_fmp: {pair_symbol} rééchantillonné en weekly. Taille: {len(df)}")

                if df.empty or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    st.error(f"analyze_fmp: Données invalides ou OHLC manquantes pour {pair_symbol} ({tf_key}).")
                    all_data_ok = False
                    break
                data_sets[tf_key] = df
            
            if not all_data_ok:
                st.write(f"DEBUG analyze_fmp: Saut de {pair_symbol}.")
                continue

            # ... (reste de la fonction analyze_forex_pairs_fmp identique pour calculs et score) ...
            data_h1, data_h4, data_d, data_w = data_sets['H1'], data_sets['H4'], data_sets['D'], data_sets['W']
            st.write(f"DEBUG analyze_fmp: Calcul des indicateurs pour {pair_symbol}.")
            
            if not all('Close' in df.columns for df in [data_h1, data_h4, data_d, data_w]):
                st.error(f"analyze_fmp: Colonne 'Close' manquante dans un DF pour {pair_symbol} avant indicateurs.")
                continue

            hma12_h1 = hma(data_h1['Close'], 12); ema20_h1 = data_h1['Close'].ewm(span=20, adjust=False).mean()
            hma12_h4 = hma(data_h4['Close'], 12); ema20_h4 = data_h4['Close'].ewm(span=20, adjust=False).mean()
            ema20_d  = data_d['Close'].ewm(span=20, adjust=False).mean(); ema50_d = data_d['Close'].ewm(span=50, adjust=False).mean()
            ema20_w  = data_w['Close'].ewm(span=20, adjust=False).mean(); ema50_w = data_w['Close'].ewm(span=50, adjust=False).mean()

            indicators = [hma12_h1, ema20_h1, hma12_h4, ema20_h4, ema20_d, ema50_d, ema20_w, ema50_w]
            if any(s is None or s.dropna().empty for s in indicators):
                st.error(f"analyze_fmp: Indicateurs avec NaN pour {pair_symbol}.")
                continue

            trend_h1 = get_trend(hma12_h1, ema20_h1)
            trend_h4 = get_trend(hma12_h4, ema20_h4)
            trend_d  = get_trend(ema20_d, ema50_d)
            trend_w  = get_trend(ema20_w, ema50_w)
            score = sum([1 if t == 'Bullish' else -1 if t == 'Bearish' else 0 for t in [trend_h1, trend_h4, trend_d, trend_w]])
            st.write(f"DEBUG analyze_fmp: Score {score} pour {pair_symbol}.")
            results.append({'Pair': f"{pair_symbol[:3]}/{pair_symbol[3:]}", 
                            'H1': trend_h1, 'H4': trend_h4, 'D': trend_d, 'W': trend_w, 'Score': score})
        except Exception as e:
            st.error(f"analyze_fmp: Erreur générale pour {pair_symbol}: {str(e)}")
            import traceback; st.error(traceback.format_exc())
            continue
            
    st.write(f"DEBUG analyze_fmp: Fin de l'analyse, {len(results)} résultats.")
    if not results: return pd.DataFrame()
    df = pd.DataFrame(results)
    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    return df
    
# ... (reste du script Streamlit pour l'interface utilisateur, qui est déjà correct) ...
    
