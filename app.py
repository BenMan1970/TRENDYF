import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import requests
import time

# --- Fonctions et Constantes (Définies en premier) ---

# Constante de la liste des paires
FOREX_PAIRS_EXTENDED = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
    'EURGBP', 'EURJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
    'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD',
    'AUDJPY', 'AUDCAD', 'AUDCHF', 'AUDNZD',
    'CADJPY', 'CADCHF',
    'CHFJPY',
    'NZDJPY', 'NZDCAD', 'NZDCHF'
]

# Fonction HMA
def hma(series, length):
    length = int(length)
    if len(series) < (length + int(math.sqrt(length)) - 1): return pd.Series([np.nan] * len(series), index=series.index, name='HMA')
    wma1_period = int(length / 2)
    sqrt_length_period = int(math.sqrt(length))
    if wma1_period < 1 or sqrt_length_period < 1: return pd.Series([np.nan] * len(series), index=series.index, name='HMA')
    wma1 = series.rolling(window=wma1_period).mean() * 2
    wma2 = series.rolling(window=length).mean()
    raw_hma = wma1 - wma2
    hma_series = raw_hma.rolling(window=sqrt_length_period).mean()
    return hma_series

# Fonction de tendance
def get_trend(fast, slow):
    if fast is None or slow is None or fast.empty or slow.empty: return 'Neutral'
    fast_last_scalar = fast.dropna().iloc[-1] if not fast.dropna().empty else np.nan
    slow_last_scalar = slow.dropna().iloc[-1] if not slow.dropna().empty else np.nan
    if pd.isna(fast_last_scalar) or pd.isna(slow_last_scalar): return 'Neutral'
    if fast_last_scalar > slow_last_scalar: return 'Bullish'
    elif fast_last_scalar < slow_last_scalar: return 'Bearish'
    else: return 'Neutral'

# Fonction de récupération des données FMP
def get_fmp_data(symbol, interval_fmp, api_key, from_date_str=None, to_date_str=None):
    base_url = "https://financialmodelingprep.com/api/v3/"
    params = {'apikey': api_key}
    data_key_in_json = None

    if interval_fmp in ['1hour', '4hour']:
        endpoint = f"historical-chart/{interval_fmp}/{symbol}"
    elif interval_fmp == '1day':
        endpoint = f"historical-price-full/{symbol}"
        data_key_in_json = "historical"
        if from_date_str: params['from'] = from_date_str
        if to_date_str: params['to'] = to_date_str
    else:
        return pd.DataFrame()

    try:
        response = requests.get(base_url + endpoint, params=params, timeout=30)
        response.raise_for_status()
        raw_data = response.json()

        if not raw_data or (isinstance(raw_data, dict) and raw_data.get('Error Message')):
            st.toast(f"API Info: {raw_data.get('Error Message', 'Réponse vide')} pour {symbol}", icon="⚠️")
            return pd.DataFrame()

        data_list = raw_data[data_key_in_json] if data_key_in_json and data_key_in_json in raw_data else raw_data
        if not isinstance(data_list, list) or not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list)
        if df.empty or 'date' not in df.columns: return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
        df.sort_index(ascending=True, inplace=True)
        return df[['Open', 'High', 'Low', 'Close']]

    except requests.exceptions.RequestException:
        st.toast(f"Erreur réseau pour {symbol}", icon="🔥")
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# Fonction d'analyse principale
# @st.cache_data(ttl=60*5) # Le cache est désactivé pour le débogage. Vous pourrez le réactiver.
def analyze_forex_pairs(api_key):
    results_internal = []
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    timeframe_params_fmp = {
        'H1': {'interval_api': '1hour', 'days_history': None},
        'H4': {'interval_api': '4hour', 'days_history': None},
        'D':  {'interval_api': '1day', 'days_history': 365},
        'W':  {'interval_api': '1day', 'days_history': 365 * 3}
    }
    
    total_pairs = len(FOREX_PAIRS_EXTENDED)
    progress_bar = st.progress(0, text=f"Analyse de 0 / {total_pairs} paires...")
    
    for i, pair_symbol in enumerate(FOREX_PAIRS_EXTENDED):
        try:
            data_sets = {}
            all_data_ok = True
            for tf_key, params in timeframe_params_fmp.items():
                from_date_str = (today - timedelta(days=params['days_history'])).strftime('%Y-%m-%d') if params['days_history'] else None
                df = get_fmp_data(pair_symbol, params['interval_api'], api_key, from_date_str, today_str)
                
                if tf_key == 'W' and not df.empty:
                    df = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()

                if df.empty:
                    all_data_ok = False
                    break
                data_sets[tf_key] = df
            
            if not all_data_ok: continue

            data_h1, data_h4, data_d, data_w = data_sets['H1'], data_sets['H4'], data_sets['D'], data_sets['W']
            trend_h1 = get_trend(hma(data_h1['Close'], 12), data_h1['Close'].ewm(span=20, adjust=False).mean())
            trend_h4 = get_trend(hma(data_h4['Close'], 12), data_h4['Close'].ewm(span=20, adjust=False).mean())
            trend_d  = get_trend(data_d['Close'].ewm(span=20, adjust=False).mean(), data_d['Close'].ewm(span=50, adjust=False).mean())
            trend_w  = get_trend(data_w['Close'].ewm(span=20, adjust=False).mean(), data_w['Close'].ewm(span=50, adjust=False).mean())
            
            score = sum([1 if t == 'Bullish' else -1 if t == 'Bearish' else 0 for t in [trend_h1, trend_h4, trend_d, trend_w]])
            results_internal.append({
                'Paire': f"{pair_symbol[:3]}/{pair_symbol[3:]}", 
                'H1': trend_h1, 'H4': trend_h4, 'D': trend_d, 'W': trend_w, 
                '_score_internal': score
            })
        except Exception:
            continue
        finally:
            progress_bar.progress((i + 1) / total_pairs, text=f"Analyse de {i+1} / {total_pairs} paires...")
    
    progress_bar.empty()
    if not results_internal: return pd.DataFrame()
    
    df_temp = pd.DataFrame(results_internal)
    df_temp.sort_values(by='_score_internal', ascending=False, inplace=True)
    return df_temp[['Paire', 'H1', 'H4', 'D', 'W']]

# --- Fonction principale de l'application (contient l'interface) ---
def main():
    st.set_page_config(layout="wide")
    st.title("Classement des Paires Forex par Tendance MTF")

    # Vérification de la clé API
    try:
        api_key = st.secrets["FMP_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("Erreur Critique: Le secret FMP_API_KEY n'est pas configuré. L'application ne peut pas fonctionner.")
        st.stop()

    # Initialisation de l'état de session
    if 'df_results' not in st.session_state:
        st.session_state.df_results = pd.DataFrame()
    if 'analysis_done_once' not in st.session_state:
        st.session_state.analysis_done_once = False

    # Logique du bouton
    if st.button("🚀 Analyser les Paires Forex"):
        with st.spinner("Analyse des paires en cours..."):
            st.session_state.df_results = analyze_forex_pairs(api_key)
            st.session_state.analysis_done_once = True

    # Logique d'affichage
    if st.session_state.analysis_done_once:
        if not st.session_state.df_results.empty:
            st.subheader("Classement des paires Forex")
            df_to_display = st.session_state.df_results.copy()
            df_to_display['Paire'] = df_to_display['Paire'].fillna('Erreur de Paire')

            def style_trends(val):
                colors = {'Bullish': '#2E7D32', 'Bearish': '#C62828', 'Neutral': '#FFD700'}
                color = colors.get(val, '')
                text_color = 'white' if val in ['Bullish', 'Bearish'] else 'black'
                return f'background-color: {color}; color: {text_color};'
            
            styled_df = df_to_display.style.map(style_trends, subset=['H1', 'H4', 'D', 'W'])
            height_dynamic = (len(df_to_display) + 1) * 35 + 3
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=height_dynamic)

            st.subheader("Résumé des Indicateurs")
            st.markdown("- **H1, H4**: Tendance basée sur HMA(12) vs EMA(20).\n- **D, W**: Tendance basée sur EMA(20) vs EMA(50).")
        else:
            st.info("L'analyse n'a produit aucun résultat. Vérifiez les logs pour plus de détails.")
    else:
        st.info("Cliquez sur 'Analyser' pour charger les données et voir le classement.")

    st.markdown("---")
    st.caption("Données via FinancialModelingPrep API.")

# --- Point d'entrée de l'application ---
if __name__ == "__main__":
    main()

