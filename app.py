import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import requests
# import time # Si votre fonction get_fmp_data réelle l'utilise pour des pauses

st.set_page_config(layout="wide")
st.title("Test avec get_fmp_data réel")

# Configuration FMP
try:
    FMP_API_KEY = st.secrets["FMP_API_KEY"]
    st.write(f"Clé API FMP lue : {FMP_API_KEY[:5]}...")
except Exception as e:
    st.error(f"Erreur de lecture du secret FMP: {e}")
    st.stop()

forex_pairs_fmp = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
st.write(f"Liste forex_pairs_fmp définie.")

# --- VOS FONCTIONS HMA et GET_TREND RÉELLES ---
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

st.write("Fonctions hma et get_trend réelles définies.")

# --- VOTRE FONCTION GET_FMP_DATA RÉELLE ---
def get_fmp_data(symbol, interval_fmp, from_date_str=None, to_date_str=None, limit=None):
    # COPIEZ ICI VOTRE FONCTION get_fmp_data COMPLÈTE
    # CELLE QUE NOUS AVONS DÉBOGUÉE ET QUI UTILISE historical-price-full pour '1day'
    # et historical-chart pour '1hour'/'4hour'.
    # Assurez-vous qu'elle a les st.write("DEBUG get_fmp_data: ...")
    pair_str_url = symbol
    st.write(f"DEBUG get_fmp_data: Appel pour {pair_str_url}, interval={interval_fmp}")
    
    base_url = "https://financialmodelingprep.com/api/v3/"
    params = {'apikey': FMP_API_KEY}
    data_key_in_json = None 

    if interval_fmp in ['1hour', '4hour']:
        endpoint = f"historical-chart/{interval_fmp}/{pair_str_url}"
    elif interval_fmp == '1day': 
        endpoint = f"historical-price-full/{pair_str_url}"
        data_key_in_json = "historical" 
        if from_date_str: params['from'] = from_date_str
        if to_date_str: params['to'] = to_date_str
    else:
        st.error(f"Intervalle FMP non géré: {interval_fmp}")
        return pd.DataFrame()

    full_url = base_url + endpoint
    
    try:
        response = requests.get(full_url, params=params)
        response.raise_for_status() 
        raw_data = response.json()
        st.write(f"DEBUG get_fmp_data: Appel API FMP à {full_url} effectué. Statut: {response.status_code}")

        if not raw_data or (isinstance(raw_data, dict) and raw_data.get('Error Message')):
            error_msg = raw_data.get('Error Message', 'Réponse vide/invalide FMP.') if isinstance(raw_data, dict) else 'Réponse vide FMP.'
            st.warning(f"Aucune donnée FMP pour {pair_str_url} ({interval_fmp}): {error_msg}")
            return pd.DataFrame()

        if data_key_in_json: 
            if data_key_in_json in raw_data and isinstance(raw_data[data_key_in_json], list):
                data_list = raw_data[data_key_in_json]
            else:
                st.warning(f"Clé '{data_key_in_json}' non trouvée ou format incorrect pour {pair_str_url} (daily).")
                return pd.DataFrame()
        else: 
            data_list = raw_data

        if not data_list:
            st.warning(f"Liste de données vide pour {pair_str_url} ({interval_fmp})")
            return pd.DataFrame()
            
        df = pd.DataFrame(data_list)
        
        if df.empty:
            st.warning(f"DataFrame vide pour {pair_str_url} ({interval_fmp})")
            return pd.DataFrame()

        if 'date' not in df.columns:
            st.error(f"Colonne 'date' manquante pour {pair_str_url} ({interval_fmp})")
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
            st.error(f"Colonnes OHLC manquantes FMP pour {pair_str_url} ({interval_fmp}).")
            return pd.DataFrame()
            
        st.write(f"DEBUG get_fmp_data: Retour pour {pair_str_url} ({interval_fmp}) avec {len(df)} lignes.")
        return df[existing_final_cols]

    except requests.exceptions.HTTPError as http_err:
        st.warning(f"Erreur HTTP FMP pour {pair_str_url} ({interval_fmp}): {http_err}. Réponse: {response.text if response else 'Pas de réponse'}")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Erreur DANS get_fmp_data pour {pair_str_url} ({interval_fmp}): {e}")
        import traceback
        st.warning(traceback.format_exc())
        return pd.DataFrame()

st.write("Fonction get_fmp_data réelle définie.")

# --- STUB POUR ANALYZE_FOREX_PAIRS_FMP ---
def analyze_forex_pairs_fmp():
    st.write(f"  [Stub] analyze_forex_pairs_fmp appelée (utilisant get_fmp_data réel)")
    results_stub = []
    # Test avec UNE SEULE PAIRE et UN SEUL TIMEFRAME pour minimiser les appels API
    pair_to_test = forex_pairs_fmp[0] # EURUSD
    interval_to_test_api = '1hour'    # Test H1
    tf_key_display = 'H1'
    
    st.write(f"    [Stub] analyze_fmp: Traitement de {pair_to_test} pour {tf_key_display}")
    
    # Appel à votre fonction get_fmp_data RÉELLE
    df_data = get_fmp_data(pair_to_test, interval_to_test_api)
    
    if not df_data.empty and 'Close' in df_data.columns:
        st.write(f"    [Stub] Données reçues pour {pair_to_test} ({tf_key_display}), taille: {len(df_data)}. Calcul indicateurs...")
        hma_val = hma(df_data['Close'], 12) 
        ema_val = df_data['Close'].ewm(span=20, adjust=False).mean() # EMA réelle
        trend = get_trend(hma_val, ema_val)
        st.write(f"    [Stub] Pour {pair_to_test}, trend {tf_key_display} (réel): {trend}")
        results_stub.append({'Pair': pair_to_test, tf_key_display: trend, 'Score': 0})
    else:
        st.write(f"    [Stub] Pas de données ou colonne 'Close' manquante pour {pair_to_test} ({tf_key_display}).")
        results_stub.append({'Pair': pair_to_test, tf_key_display: 'ErrorData', 'Score': 0})
        
    return pd.DataFrame(results_stub)

st.write("--- Fin du test avec get_fmp_data réel ---")

try:
    st.write("Test d'appel de analyze_forex_pairs_fmp (stub, get_fmp_data réel)...")
    test_df = analyze_forex_pairs_fmp() # Cet appel va maintenant déclencher un vrai appel API via get_fmp_data
    st.write("analyze_forex_pairs_fmp (stub, get_fmp_data réel) appelée avec succès. Résultat :")
    st.write(test_df)
except Exception as e_call:
    st.error(f"Erreur lors de l'appel de analyze_forex_pairs_fmp (stub, get_fmp_data réel): {e_call}")
    import traceback
    st.error(traceback.format_exc())
    
