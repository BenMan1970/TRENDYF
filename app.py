# TOUT EN HAUT DU FICHIER APP.PY
import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import requests # Pour les appels API à FMP
import time     # Pour d'éventuelles petites pauses si besoin (moins critique avec FMP)

# PREMIÈRE COMMANDE STREAMLIT
st.set_page_config(layout="wide")

st.write("Début du script - Imports OK, Page Config OK") # LOG

# Configuration FMP (depuis st.secrets)
st.write("Tentative de lecture des secrets FMP...") # LOG
try:
    FMP_API_KEY = st.secrets["FMP_API_KEY"]
    st.write("Secret FMP_API_KEY lu avec succès.") # LOG
except KeyError:
    st.error("Erreur: Le secret FMP_API_KEY n'est pas configuré.")
    st.stop()
except Exception as e:
    st.error(f"Erreur inattendue lors de la lecture des secrets FMP: {e}")
    st.stop()

# Liste des paires Forex à analyser (format FMP : EURUSD)
forex_pairs_fmp = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']

st.write("Définitions des fonctions et listes de paires OK.") # LOG

# --- Fonctions de calcul et d'analyse (identiques aux versions précédentes) ---
def hma(series, length):
    length = int(length)
    min_points_needed = length + int(math.sqrt(length)) -1
    if len(series) < min_points_needed: return pd.Series([np.nan] * len(series), index=series.index, name='HMA')
    wma1_period = int(length / 2)
    sqrt_length_period = int(math.sqrt(length))
    if wma1_period < 1 or length < 1 or sqrt_length_period < 1: return pd.Series([np.nan] * len(series), index=series.index, name='HMA')
    wma1 = series.rolling(window=wma1_period, min_periods=wma1_period).mean() * 2
    wma2 = series.rolling(window=length, min_periods=length).mean()
    raw_hma = wma1 - wma2
    hma_series = raw_hma.rolling(window=sqrt_length_period, min_periods=sqrt_length_period).mean()
    return hma_series

def get_trend(fast, slow):
    if fast is None or slow is None or fast.empty or slow.empty: return 'Neutral'
    fast_dn, slow_dn = fast.dropna(), slow.dropna()
    fast_last_scalar, slow_last_scalar = np.nan, np.nan
    if not fast_dn.empty:
        val = fast_dn.iloc[-1]
        if isinstance(val, pd.Series): fast_last_scalar = val.item() if len(val) == 1 else np.nan
        else: fast_last_scalar = val
    if not slow_dn.empty:
        val = slow_dn.iloc[-1]
        if isinstance(val, pd.Series): slow_last_scalar = val.item() if len(val) == 1 else np.nan
        else: slow_last_scalar = val
    if pd.isna(fast_last_scalar) or pd.isna(slow_last_scalar): return 'Neutral'
    if fast_last_scalar > slow_last_scalar: return 'Bullish'
    elif fast_last_scalar < slow_last_scalar: return 'Bearish'
    else: return 'Neutral'

# Fonction pour récupérer les données FMP
def get_fmp_data(symbol, interval_fmp, from_date_str=None, to_date_str=None, limit=None):
    """
    Récupère les données de FMP.
    Pour les données intraday (1hour, 4hour), FMP utilise un endpoint différent.
    Pour les données daily/weekly, FMP utilise l'endpoint 'historical-chart'.
    'interval_fmp' peut être '1hour', '4hour', '1day', '1week'.
    """
    pair_str_url = symbol # Pour FMP, le symbole est direct ex: EURUSD
    st.write(f"DEBUG get_fmp_data: Appel pour {pair_str_url}, interval={interval_fmp}")
    
    base_url = "https://financialmodelingprep.com/api/v3/"
    params = {'apikey': FMP_API_KEY}

    if interval_fmp in ['1hour', '4hour']:
        # Endpoint pour données intraday Forex
        # https://financialmodelingprep.com/api/v3/historical-chart/1hour/EURUSD?apikey=YOUR_API_KEY
        # https://financialmodelingprep.com/api/v3/historical-chart/4hour/EURUSD?apikey=YOUR_API_KEY
        # FMP intraday Forex peut retourner un nombre limité de points par défaut, on n'utilise pas from/to ici.
        # Si 'limit' est fourni, il pourrait être utilisé, mais la doc est moins claire pour les intraday Forex via cet endpoint.
        # Pour l'instant, on se fie au comportement par défaut pour intraday qui est généralement suffisant pour HMA/EMA.
        endpoint = f"historical-chart/{interval_fmp}/{pair_str_url}"
        if limit: # FMP ne semble pas supporter 'limit' pour cet endpoint spécifique, mais on le garde au cas où
            # params['limit'] = limit # La doc FMP n'indique pas de param 'limit' pour cet endpoint.
            # L'endpoint intraday retourne généralement les 30-90 derniers jours pour H1.
            pass 
    elif interval_fmp in ['1day', '1week']:
        # Endpoint pour données historiques journalières/hebdomadaires
        # https://financialmodelingprep.com/api/v3/historical-price-full/EURUSD?apikey=YOUR_API_KEY (pour daily)
        # Pour weekly, on demande daily et on rééchantillonne, ou on voit si un endpoint weekly existe.
        # L'endpoint "historical-chart/{timeseries}/{TICKER}" est plus général.
        # 1day: https://financialmodelingprep.com/api/v3/historical-chart/daily/EURUSD (n'existe pas, il faut 1day)
        # FMP utilise 'daily' comme timeseries pour les données journalières dans l'URL.
        # Pour weekly, on va prendre daily et resample.
        if interval_fmp == '1day':
            endpoint = f"historical-chart/daily/{pair_str_url}" # FMP utilise 'daily' ici.
        elif interval_fmp == '1week':
            # Pour weekly, on récupère les données journalières sur une plus longue période et on rééchantillonne
            endpoint = f"historical-chart/daily/{pair_str_url}" # Récupérer daily
            # On ajustera from_date_str pour weekly plus bas si besoin pour avoir assez de données
        else: # Ne devrait pas arriver
            st.error(f"Intervalle FMP non supporté : {interval_fmp}")
            return pd.DataFrame()
        
        if from_date_str: params['from'] = from_date_str
        if to_date_str: params['to'] = to_date_str
        # 'limit' est supporté pour historical-price-full, mais pas toujours pour historical-chart
    else:
        st.error(f"Intervalle FMP non géré : {interval_fmp}")
        return pd.DataFrame()

    full_url = base_url + endpoint
    
    try:
        response = requests.get(full_url, params=params)
        response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP (4XX, 5XX)
        data = response.json()
        st.write(f"DEBUG get_fmp_data: Appel API FMP à {full_url} effectué. Statut: {response.status_code}")

        if not data or (isinstance(data, dict) and data.get('Error Message')):
            error_msg = data.get('Error Message', 'Réponse vide ou invalide de FMP.') if isinstance(data, dict) else 'Réponse vide de FMP.'
            st.warning(f"Aucune donnée FMP (ou erreur) pour {pair_str_url} ({interval_fmp}): {error_msg}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        
        if df.empty:
            st.warning(f"DataFrame vide après conversion JSON pour {pair_str_url} ({interval_fmp})")
            return pd.DataFrame()

        # FMP retourne les colonnes 'date', 'open', 'high', 'low', 'close', 'volume'
        # L'index doit être 'date' et converti en datetime
        if 'date' not in df.columns:
            st.error(f"Colonne 'date' manquante dans les données FMP pour {pair_str_url} ({interval_fmp})")
            return pd.DataFrame()
            
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Renommer les colonnes
        rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        df.rename(columns=rename_map, inplace=True)
        df.sort_index(ascending=True, inplace=True)

        # Si on a demandé '1week', rééchantillonner les données journalières
        if interval_fmp == '1week':
            if not df.empty:
                # 'W-FRI' pour que la semaine se termine le vendredi, commun pour les données financières
                # ou simplement 'W' pour la fin de semaine par défaut de Pandas.
                # Le 'last()' prend la dernière valeur de la semaine pour OHLCV.
                # Pour Open, on voudrait 'first()', High 'max()', Low 'min()', Close 'last()', Volume 'sum()'.
                agg_funcs = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
                if 'Volume' in df.columns: agg_funcs['Volume'] = 'sum'
                
                df_weekly = df.resample('W-FRI').agg(agg_funcs)
                df = df_weekly.dropna() # Enlever les semaines sans données complètes
                st.write(f"DEBUG get_fmp_data: Rééchantillonné en hebdomadaire pour {pair_str_url}. Taille: {len(df)}")
            else:
                st.warning(f"Impossible de rééchantillonner en weekly, DataFrame journalier vide pour {pair_str_url}")
                return pd.DataFrame()

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

def analyze_forex_pairs_fmp():
    st.write("DEBUG analyze_fmp: Début de l'analyse.")
    results = []
    today_str = datetime.now().strftime('%Y-%m-%d')

    # FMP a des endpoints pour 1hour, 4hour, et daily. Weekly sera resamplé.
    # Pour intraday, FMP retourne un historique limité (ex: 30-90 jours pour H1).
    # Pour daily/weekly, on peut spécifier 'from' et 'to' ou se fier au 'limit' si supporté.
    timeframe_params_fmp = {
        'H1': {'interval': '1hour', 'days_for_weekly_resample': None}, 
        'H4': {'interval': '4hour', 'days_for_weekly_resample': None},
        'D':  {'interval': '1day',  'days_for_weekly_resample': None},
        'W':  {'interval': '1week', 'days_for_weekly_resample': 3*365} # Pour weekly, on prend ~3 ans de daily
    }

    for pair_symbol in forex_pairs_fmp:
        st.write(f"DEBUG analyze_fmp: Traitement de {pair_symbol}.")
        try:
            data_sets = {}
            all_data_ok = True
            for tf_key, params in timeframe_params_fmp.items():
                from_date_req = None
                if params['days_for_weekly_resample'] and tf_key == 'W': # Cas spécial pour weekly
                    from_date_req = (datetime.now() - timedelta(days=params['days_for_weekly_resample'])).strftime('%Y-%m-%d')
                    actual_interval_to_fetch = '1day' # On fetch daily pour resampler en weekly
                elif tf_key == 'D': # Pour daily, on peut prendre un historique plus long
                    from_date_req = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d') # 1 an de daily
                    actual_interval_to_fetch = params['interval']
                else: # H1, H4
                    actual_interval_to_fetch = params['interval']
                
                st.write(f"DEBUG analyze_fmp: Récupération {tf_key} (interval FMP: {actual_interval_to_fetch}) pour {pair_symbol}.")
                df = get_fmp_data(
                    symbol=pair_symbol,
                    interval_fmp=actual_interval_to_fetch,
                    from_date_str=from_date_req,
                    to_date_str=today_str if from_date_req else None # 'to' seulement si 'from' est utilisé
                )
                
                # Si on a récupéré daily pour faire du weekly, et que l'intervalle original était '1week'
                if params['interval'] == '1week' and actual_interval_to_fetch == '1day' and not df.empty:
                    agg_funcs = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
                    if 'Volume' in df.columns: agg_funcs['Volume'] = 'sum'
                    df_weekly = df.resample('W-FRI').agg(agg_funcs)
                    df = df_weekly.dropna()
                    st.write(f"DEBUG analyze_fmp: {pair_symbol} rééchantillonné en weekly. Taille: {len(df)}")


                if df.empty or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    st.error(f"analyze_fmp: Données invalides ou OHLC manquantes pour {pair_symbol} ({tf_key}).")
                    all_data_ok = False
                    break
                data_sets[tf_key] = df
            
            if not all_data_ok:
                st.write(f"DEBUG analyze_fmp: Saut de {pair_symbol}.")
                continue

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

# --- Interface Streamlit ---
st.title("Classement des Paires Forex par Tendance MTF (via FMP)")

if 'data_loaded_fmp' not in st.session_state: st.session_state.data_loaded_fmp = False
if 'df_results_fmp' not in st.session_state: st.session_state.df_results_fmp = pd.DataFrame()

bouton_clique_fmp = st.button("Actualiser et Analyser (FMP)")

if bouton_clique_fmp:
    st.write("DEBUG Bouton FMP cliqué.")
    st.session_state.data_loaded_fmp = False
    st.session_state.df_results_fmp = pd.DataFrame()
    with st.spinner("Analyse FMP en cours..."):
        df_temp_fmp = analyze_forex_pairs_fmp()
    
    if df_temp_fmp is not None and not df_temp_fmp.empty:
        st.session_state.df_results_fmp = df_temp_fmp
        st.session_state.data_loaded_fmp = True
        st.write("DEBUG Résultats FMP stockés.")
    else:
        st.warning("Aucun résultat concluant (FMP).")

if st.session_state.data_loaded_fmp and not st.session_state.df_results_fmp.empty:
    st.write("DEBUG Affichage des résultats FMP.")
    df_to_display_fmp = st.session_state.df_results_fmp
    
    st.subheader("Classement des paires Forex")
    trend_cols = ['H1', 'H4', 'D', 'W']
    def style_trends(val):
        if val == 'Bullish': return 'background-color: mediumseagreen; color: white;'
        elif val == 'Bearish': return 'background-color: indianred; color: white;'
        elif val == 'Neutral': return 'background-color: khaki; color: black;'
        else: return 'background-color: whitesmoke; color: black;'
    
    existing_trend_cols = [col for col in trend_cols if col in df_to_display_fmp.columns]
    if existing_trend_cols:
        styled_df_fmp = df_to_display_fmp.style.applymap(style_trends, subset=pd.IndexSlice[:, existing_trend_cols])
        st.dataframe(styled_df_fmp, use_container_width=True)
    else:
        st.dataframe(df_to_display_fmp, use_container_width=True)


    st.subheader("Résumé des Indicateurs")
    st.markdown("""
    - **H1, H4**: Tendance basée sur HMA(12) vs EMA(20).
    - **D, W**: Tendance basée sur EMA(20) vs EMA(50).
    - **Score**: Somme des tendances (+1 Bullish, -1 Bearish, 0 Neutral).
    """)
elif bouton_clique_fmp: pass
else: st.info("Cliquez sur le bouton pour lancer l'analyse FMP.")
