import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import requests
import time # Potentiellement utilisé par analyze_forex_pairs_fmp pour des pauses si nécessaire

st.set_page_config(layout="wide")
st.title("Application Complète avec FMP")

# Configuration FMP
try:
    FMP_API_KEY = st.secrets["FMP_API_KEY"]
    st.write(f"DEBUG Clé API FMP lue.")
except Exception as e:
    st.error(f"Erreur de lecture du secret FMP: {e}")
    st.stop()

forex_pairs_fmp = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']

# --- VOS FONCTIONS RÉELLES : hma, get_trend, get_fmp_data ---
def hma(series, length): # VOTRE CODE hma COMPLET ICI
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

def get_trend(fast, slow): # VOTRE CODE get_trend COMPLET ICI
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

def get_fmp_data(symbol, interval_fmp, from_date_str=None, to_date_str=None, limit=None):
    # VOTRE FONCTION get_fmp_data RÉELLE et COMPLÈTE (celle avec les logs DEBUG détaillés et le timeout)
    pair_str_url = symbol
    st.write(f"===> get_fmp_data: IN - Symbol={pair_str_url}, Interval={interval_fmp}, From={from_date_str}, To={to_date_str}")
    base_url = "https://financialmodelingprep.com/api/v3/"
    params = {'apikey': FMP_API_KEY}
    data_key_in_json = None 
    if interval_fmp in ['1hour', '4hour']:
        endpoint = f"historical-chart/{interval_fmp}/{pair_str_url}"
        st.write(f"     get_fmp_data: Intraday endpoint: {endpoint}")
    elif interval_fmp == '1day': 
        endpoint = f"historical-price-full/{pair_str_url}"
        data_key_in_json = "historical" 
        if from_date_str: params['from'] = from_date_str
        if to_date_str: params['to'] = to_date_str
        st.write(f"     get_fmp_data: Daily endpoint: {endpoint} with params: {params}")
    else:
        st.error(f"     get_fmp_data: Intervalle FMP non géré: {interval_fmp}")
        return pd.DataFrame()
    full_url = base_url + endpoint
    try:
        st.write(f"     get_fmp_data: Préparation de l'appel à {full_url} avec params: {params}")
        response = requests.get(full_url, params=params, timeout=25) 
        st.write(f"     get_fmp_data: Appel API terminé. Statut: {response.status_code}. URL: {response.url}")
        response.raise_for_status() 
        st.write(f"     get_fmp_data: Tentative de conversion en JSON...")
        raw_data = response.json()
        st.write(f"     get_fmp_data: Conversion JSON réussie. Type: {type(raw_data)}")
        if not raw_data or (isinstance(raw_data, dict) and raw_data.get('Error Message')):
            error_msg = raw_data.get('Error Message', 'Rép. vide/invalide FMP.') if isinstance(raw_data, dict) else 'Rép. vide FMP.'
            st.warning(f"     get_fmp_data: Aucune donnée {pair_str_url} ({interval_fmp}): {error_msg}")
            return pd.DataFrame()
        if data_key_in_json: 
            st.write(f"     get_fmp_data: Extraction de '{data_key_in_json}'...")
            if data_key_in_json in raw_data and isinstance(raw_data[data_key_in_json], list):
                data_list = raw_data[data_key_in_json]
                st.write(f"     get_fmp_data: Données extraites, {len(data_list)} éls.")
            else:
                st.warning(f"     get_fmp_data: Clé '{data_key_in_json}' non trouvée/format incorrect {pair_str_url}. raw_data: {str(raw_data)[:200]}")
                return pd.DataFrame()
        else: 
            st.write(f"     get_fmp_data: Utilisation directe de raw_data (intraday)...")
            data_list = raw_data
            st.write(f"     get_fmp_data: Données intraday prêtes, {len(data_list)} éls.")
        if not data_list:
            st.warning(f"     get_fmp_data: Liste de données vide après extraction {pair_str_url} ({interval_fmp})")
            return pd.DataFrame()
        st.write(f"     get_fmp_data: Tentative création DataFrame Pandas...")    
        df = pd.DataFrame(data_list)
        st.write(f"     get_fmp_data: DataFrame créé. Taille: {df.shape}")
        if df.empty:
            st.warning(f"     get_fmp_data: DataFrame vide {pair_str_url} ({interval_fmp})")
            return pd.DataFrame()
        if 'date' not in df.columns:
            st.error(f"     get_fmp_data: Colonne 'date' manquante {pair_str_url} ({interval_fmp}). Cols: {df.columns.tolist()}")
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
            st.error(f"     get_fmp_data: Cols OHLC manquantes {pair_str_url} ({interval_fmp}). Actuelles: {df.columns.tolist()}")
            return pd.DataFrame()
        st.write(f"<=== get_fmp_data: OUT - Retour pour {pair_str_url} ({interval_fmp}) avec {len(df)} lignes.")
        return df[existing_final_cols]
    except requests.exceptions.Timeout:
        st.warning(f"     get_fmp_data: Timeout API pour {pair_str_url} ({interval_fmp}) URL {full_url}")
        return pd.DataFrame()
    except requests.exceptions.HTTPError as http_err:
        st.warning(f"     get_fmp_data: Erreur HTTP FMP {pair_str_url} ({interval_fmp}): {http_err}. URL: {full_url}. Rép: {response.text if 'response' in locals() and response is not None else 'N/A'}")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"     get_fmp_data: Erreur DANS get_fmp_data {pair_str_url} ({interval_fmp}): {e}")
        import traceback
        st.warning(traceback.format_exc())
        return pd.DataFrame()

# --- VOTRE FONCTION ANALYZE_FOREX_PAIRS_FMP RÉELLE ---
def analyze_forex_pairs_fmp():
    # COPIEZ ICI VOTRE FONCTION analyze_forex_pairs_fmp COMPLÈTE
    # Celle qui boucle sur toutes les paires et tous les timeframes,
    # appelle get_fmp_data, gère le resample pour Weekly, calcule les indicateurs.
    # Assurez-vous qu'elle a les st.write("DEBUG analyze_fmp: ...")
    st.write("DEBUG analyze_fmp: Début de l'analyse.")
    results = []
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    timeframe_params_fmp = {
        'H1': {'interval_api': '1hour', 'days_history_for_daily_weekly': None}, 
        'H4': {'interval_api': '4hour', 'days_history_for_daily_weekly': None},
        'D':  {'interval_api': '1day',  'days_history_for_daily_weekly': 365 * 1}, 
        'W':  {'interval_api': '1day',  'days_history_for_daily_weekly': 365 * 3}
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
                actual_interval_to_fetch = params['interval_api']
                st.write(f"DEBUG analyze_fmp: Récupération {tf_key} (API: {actual_interval_to_fetch}) pour {pair_symbol}.")
                df = get_fmp_data(
                    symbol=pair_symbol,
                    interval_fmp=actual_interval_to_fetch,
                    from_date_str=from_date_req_str,
                    to_date_str=today_str if from_date_req_str else None
                )
                if tf_key == 'W' and not df.empty: # Resample si c'est pour Weekly
                    agg_funcs = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
                    if 'Volume' in df.columns: agg_funcs['Volume'] = 'sum'
                    df_weekly = df.resample('W-FRI').agg(agg_funcs) 
                    df = df_weekly.dropna() 
                    st.write(f"DEBUG analyze_fmp: {pair_symbol} rééchantillonné en weekly. Taille: {len(df)}")
                if df.empty or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    st.error(f"analyze_fmp: Données invalides/OHLC manquantes pour {pair_symbol} ({tf_key}).")
                    all_data_ok = False
                    break
                data_sets[tf_key] = df
            if not all_data_ok:
                st.write(f"DEBUG analyze_fmp: Saut de {pair_symbol}.")
                continue
            data_h1, data_h4, data_d, data_w = data_sets['H1'], data_sets['H4'], data_sets['D'], data_sets['W']
            st.write(f"DEBUG analyze_fmp: Calcul indicateurs pour {pair_symbol}.")
            if not all('Close' in df.columns for df in [data_h1, data_h4, data_d, data_w]):
                st.error(f"analyze_fmp: Col 'Close' manquante {pair_symbol} avant indicateurs.")
                continue
            hma12_h1 = hma(data_h1['Close'], 12); ema20_h1 = data_h1['Close'].ewm(span=20, adjust=False).mean()
            hma12_h4 = hma(data_h4['Close'], 12); ema20_h4 = data_h4['Close'].ewm(span=20, adjust=False).mean()
            ema20_d  = data_d['Close'].ewm(span=20, adjust=False).mean(); ema50_d = data_d['Close'].ewm(span=50, adjust=False).mean()
            ema20_w  = data_w['Close'].ewm(span=20, adjust=False).mean(); ema50_w = data_w['Close'].ewm(span=50, adjust=False).mean()
            indicators = [hma12_h1, ema20_h1, hma12_h4, ema20_h4, ema20_d, ema50_d, ema20_w, ema50_w]
            if any(s is None or s.dropna().empty for s in indicators):
                st.error(f"analyze_fmp: Indicateurs avec NaN pour {pair_symbol}.")
                continue
            trend_h1 = get_trend(hma12_h1, ema20_h1); trend_h4 = get_trend(hma12_h4, ema20_h4)
            trend_d  = get_trend(ema20_d, ema50_d); trend_w  = get_trend(ema20_w, ema50_w)
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

st.write("Fonctions réelles (hma, get_trend, get_fmp_data, analyze_forex_pairs_fmp) définies.")

# --- Interface Streamlit ---
# (Votre code d'interface Streamlit avec le bouton et l'affichage des résultats,
#  qui appelle analyze_forex_pairs_fmp, est déjà correct et peut être gardé tel quel.)
if 'data_loaded_fmp' not in st.session_state: st.session_state.data_loaded_fmp = False
if 'df_results_fmp' not in st.session_state: st.session_state.df_results_fmp = pd.DataFrame() 

bouton_clique_fmp = st.button("Actualiser et Analyser (FMP)")
st.write(f"DEBUG État du bouton avant le if: {bouton_clique_fmp}")

if bouton_clique_fmp:
    st.write("DEBUG Bouton FMP cliqué.")
    st.session_state.data_loaded_fmp = False 
    st.session_state.df_results_fmp = pd.DataFrame() 
    with st.spinner("Analyse FMP en cours..."):
        st.write("DEBUG Spinner activé, appel de analyze_forex_pairs_fmp...")
        df_temp_fmp = analyze_forex_pairs_fmp() # APPEL À LA FONCTION RÉELLE
    st.write(f"DEBUG analyze_forex_pairs_fmp a retourné : {type(df_temp_fmp)}")
    if df_temp_fmp is not None and not df_temp_fmp.empty:
        st.session_state.df_results_fmp = df_temp_fmp
        st.session_state.data_loaded_fmp = True
        st.write("DEBUG Résultats FMP stockés.")
    else:
        st.warning("Aucun résultat concluant (FMP).")
        st.session_state.data_loaded_fmp = False 
# ... (reste de la logique d'affichage comme dans la version complète précédente) ...
if st.session_state.data_loaded_fmp and isinstance(st.session_state.df_results_fmp, pd.DataFrame) and not st.session_state.df_results_fmp.empty:
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
elif bouton_clique_fmp and (not st.session_state.data_loaded_fmp or (isinstance(st.session_state.df_results_fmp, pd.DataFrame) and st.session_state.df_results_fmp.empty)):
    st.write("DEBUG: Bouton cliqué, mais pas de données valides à afficher.")
elif not bouton_clique_fmp:
    st.info("Cliquez sur le bouton pour lancer l'analyse FMP.")
 
    
