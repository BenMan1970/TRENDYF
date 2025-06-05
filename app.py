# TOUT EN HAUT DU FICHIER APP.PY
import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import time # Pour les pauses éventuelles

# Essayer d'importer AlphaVantage, gérer l'erreur si l'import échoue
try:
    from alpha_vantage.foreignexchange import ForeignExchange
    from alpha_vantage.timeseries import TimeSeries # Pour Daily et Weekly
except ImportError as e:
    ALPHAVANTAGE_IMPORT_ERROR = f"ERREUR D'IMPORT: Impossible d'importer alpha_vantage: {e}. Assurez-vous que 'alpha-vantage' est dans requirements.txt et installé."
else:
    ALPHAVANTAGE_IMPORT_ERROR = None

# PREMIÈRE COMMANDE STREAMLIT
st.set_page_config(layout="wide")

# Afficher l'erreur d'import ici si elle a eu lieu
if ALPHAVANTAGE_IMPORT_ERROR:
    st.error(ALPHAVANTAGE_IMPORT_ERROR)
    st.stop()

st.write("Début du script - Imports OK, Page Config OK") # LOG

# Configuration Alpha Vantage (depuis st.secrets)
st.write("Tentative de lecture des secrets Alpha Vantage...") # LOG
try:
    ALPHAVANTAGE_API_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]
    st.write("Secret ALPHAVANTAGE_API_KEY lu avec succès.") # LOG
except KeyError:
    st.error("Erreur: Le secret ALPHAVANTAGE_API_KEY n'est pas configuré.")
    st.stop()
except Exception as e:
    st.error(f"Erreur inattendue lors de la lecture des secrets Alpha Vantage: {e}")
    st.stop()

# Initialisation des clients Alpha Vantage
st.write("Tentative d'initialisation des clients Alpha Vantage...") # LOG
try:
    # Alpha Vantage retourne les données en JSON, pandas pour le formatage
    fx = ForeignExchange(key=ALPHAVANTAGE_API_KEY, output_format='pandas')
    ts_client = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format='pandas') # Pour Daily/Weekly Forex qui sont sous TimeSeries
    st.write("Clients Alpha Vantage initialisés avec succès.") # LOG
except Exception as e:
    st.error(f"Erreur lors de l'initialisation des clients Alpha Vantage: {e}")
    import traceback
    st.error(traceback.format_exc())
    st.stop()

# Liste des paires Forex à analyser (format Alpha Vantage: FROM_CURRENCY, TO_CURRENCY)
# Et pour les time series, le symbole est FROM_CURRENCY + TO_CURRENCY (ex: EURUSD)
forex_pairs_alphavantage = [
    {'from': 'EUR', 'to': 'USD', 'symbol_ts': 'EURUSD'},
    {'from': 'GBP', 'to': 'USD', 'symbol_ts': 'GBPUSD'},
    {'from': 'USD', 'to': 'JPY', 'symbol_ts': 'USDJPY'},
    {'from': 'AUD', 'to': 'USD', 'symbol_ts': 'AUDUSD'},
    {'from': 'USD', 'to': 'CAD', 'symbol_ts': 'USDCAD'},
    {'from': 'USD', 'to': 'CHF', 'symbol_ts': 'USDCHF'},
    {'from': 'NZD', 'to': 'USD', 'symbol_ts': 'NZDUSD'}
]

st.write("Définitions des fonctions et listes de paires OK.") # LOG

# --- Fonctions de calcul et d'analyse ---
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

def get_alphavantage_data(pair_info, interval_av):
    pair_str = f"{pair_info['from']}/{pair_info['to']}"
    st.write(f"DEBUG get_alphavantage_data: Appel pour {pair_str}, interval={interval_av}")
    df = None
    meta_data = None
    try:
        if interval_av == '60min': # H1
            # Pour FX_INTRADAY, on a besoin de 'from_symbol' et 'to_symbol'
            df, meta_data = fx.get_currency_exchange_intraday(
                from_symbol=pair_info['from'],
                to_symbol=pair_info['to'],
                interval=interval_av,
                outputsize='full' # ou 'compact' pour 100 points
            )
        elif interval_av == 'daily': # D
             # Les endpoints FX_DAILY de Alpha Vantage sont un peu étranges.
             # On va utiliser get_daily_adjusted de TimeSeries, en espérant que le symbole simple (ex: EURUSD) fonctionne.
             # Alternativement, il y a fx.get_currency_exchange_daily mais il demande from/to.
             # La documentation suggère que FX_DAILY est sous TimeSeries.
            df, meta_data = ts_client.get_daily_adjusted(symbol=pair_info['symbol_ts'], outputsize='full')

        elif interval_av == 'weekly': # W
            df, meta_data = ts_client.get_weekly_adjusted(symbol=pair_info['symbol_ts'], outputsize='full')
        
        st.write(f"DEBUG get_alphavantage_data: Appel API pour {pair_str} ({interval_av}) effectué.")

        if df is None or df.empty:
            st.warning(f"Aucune donnée Alpha Vantage retournée pour {pair_str} ({interval_av})")
            return pd.DataFrame()

        # Alpha Vantage nomme ses colonnes différemment, ex: '1. open', '4. close'
        # Et l'index est 'date'
        rename_map = {}
        for col in df.columns:
            if 'open' in col: rename_map[col] = 'Open'
            elif 'high' in col: rename_map[col] = 'High'
            elif 'low' in col: rename_map[col] = 'Low'
            elif 'close' in col: rename_map[col] = 'Close'
            elif 'volume' in col: rename_map[col] = 'Volume'
        
        df.rename(columns=rename_map, inplace=True)
        df.index = pd.to_datetime(df.index) # S'assurer que l'index est Datetime
        df.sort_index(ascending=True, inplace=True) # Le plus ancien en premier

        st.write(f"DEBUG get_alphavantage_data: Données formatées pour {pair_str} ({interval_av}). Colonnes: {df.columns.tolist()}")

        # Sélectionner uniquement les colonnes nécessaires et existantes
        final_cols = ['Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns:
            final_cols.append('Volume')
        
        existing_final_cols = [c for c in final_cols if c in df.columns]
        
        if not all(c in existing_final_cols for c in ['Open', 'High', 'Low', 'Close']):
            st.error(f"Colonnes OHLC manquantes après traitement pour {pair_str} ({interval_av}).")
            return pd.DataFrame()

        return df[existing_final_cols]

    except ValueError as ve: # Souvent levé par Alpha Vantage pour limites API ou symbole incorrect
        st.warning(f"Erreur Alpha Vantage (probablement limite API ou symbole) pour {pair_str} ({interval_av}): {ve}")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Erreur DANS get_alphavantage_data pour {pair_str} ({interval_av}): {e}")
        import traceback
        st.warning(traceback.format_exc())
        return pd.DataFrame()

def analyze_forex_pairs_alphavantage():
    st.write("DEBUG analyze_alphavantage: Début de l'analyse.")
    results = []
    # Timeframes : H1, D, W. H4 est omis pour Alpha Vantage (ou nécessite agrégation)
    timeframe_map_av = {
        'H1': '60min',
        'D': 'daily',
        'W': 'weekly'
    }
    
    API_CALLS_PER_MINUTE_LIMIT_AV = 5 # Limite typique d'Alpha Vantage, à ajuster si vous connaissez la vôtre
    api_calls_this_minute = 0
    last_api_call_time = time.time()

    for i, pair_config in enumerate(forex_pairs_alphavantage):
        pair_name = f"{pair_config['from']}/{pair_config['to']}"
        st.write(f"DEBUG analyze_alphavantage: Traitement de {pair_name} ({i+1}/{len(forex_pairs_alphavantage)}).")
        
        data_sets = {}
        all_data_fetched_successfully = True
        
        for tf_key_display, av_interval_key in timeframe_map_av.items():
            current_time = time.time()
            if api_calls_this_minute >= API_CALLS_PER_MINUTE_LIMIT_AV and (current_time - last_api_call_time < 60) :
                wait_time = 60 - (current_time - last_api_call_time) + 5 # Attendre le reste de la minute + 5s marge
                st.write(f"DEBUG analyze_alphavantage: Limite API AV atteinte. Pause de {wait_time:.2f} secondes...")
                time.sleep(wait_time)
                api_calls_this_minute = 0
            
            st.write(f"DEBUG analyze_alphavantage: Récupération {tf_key_display} (AV: {av_interval_key}) pour {pair_name}.")
            df = get_alphavantage_data(pair_config, av_interval_key)
            api_calls_this_minute += 1
            last_api_call_time = time.time()

            if df.empty or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                st.error(f"analyze_alphavantage: Données invalides pour {pair_name} ({tf_key_display}).")
                all_data_fetched_successfully = False
                break
            data_sets[tf_key_display] = df

        if not all_data_fetched_successfully:
            st.write(f"DEBUG analyze_alphavantage: Saut de {pair_name}.")
            api_calls_this_minute = 0 # Réinitialiser pour la prochaine paire si celle-ci a échoué tôt
            continue

        data_h1, data_d, data_w = data_sets['H1'], data_sets['D'], data_sets['W']
        st.write(f"DEBUG analyze_alphavantage: Calcul des indicateurs pour {pair_name}.")

        # HMA pour H1, EMA pour D et W
        hma12_h1 = hma(data_h1['Close'], 12)
        ema20_h1 = data_h1['Close'].ewm(span=20, adjust=False).mean()
        
        ema20_d = data_d['Close'].ewm(span=20, adjust=False).mean()
        ema50_d = data_d['Close'].ewm(span=50, adjust=False).mean()
        
        ema20_w = data_w['Close'].ewm(span=20, adjust=False).mean()
        ema50_w = data_w['Close'].ewm(span=50, adjust=False).mean()

        indicators = [hma12_h1, ema20_h1, ema20_d, ema50_d, ema20_w, ema50_w]
        if any(s is None or s.dropna().empty for s in indicators):
            st.error(f"analyze_alphavantage: Indicateurs avec NaN pour {pair_name}.")
            continue

        trend_h1 = get_trend(hma12_h1, ema20_h1)
        trend_d = get_trend(ema20_d, ema50_d)
        trend_w = get_trend(ema20_w, ema50_w)

        # Score basé sur H1, D, W (pas de H4 ici)
        score = sum([1 if t == 'Bullish' else -1 if t == 'Bearish' else 0 for t in [trend_h1, trend_d, trend_w]])
        st.write(f"DEBUG analyze_alphavantage: Score {score} pour {pair_name}.")

        results.append({
            'Pair': pair_name,
            'H1': trend_h1,
            # 'H4': 'N/A', # Indiquer que H4 n'est pas applicable avec cette source
            'D': trend_d,
            'W': trend_w,
            'Score': score
        })
        
    st.write(f"DEBUG analyze_alphavantage: Fin de l'analyse, {len(results)} résultats.")
    if not results: return pd.DataFrame()
    df = pd.DataFrame(results)
    # S'assurer que les colonnes pour le style existent avant de trier
    # Si on enlève H4, il faut ajuster la liste des colonnes de style aussi
    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    return df


# --- Interface Streamlit ---
st.title("Classement des Paires Forex par Tendance MTF (via Alpha Vantage)")
st.write("Analyse sur H1, D, W. H4 non disponible directement via Alpha Vantage.")

if 'data_loaded_av' not in st.session_state: st.session_state.data_loaded_av = False
if 'df_results_av' not in st.session_state: st.session_state.df_results_av = pd.DataFrame()

bouton_clique_av = st.button("Actualiser et Analyser (Alpha Vantage)")

if bouton_clique_av:
    st.write("DEBUG Bouton Alpha Vantage cliqué.")
    st.session_state.data_loaded_av = False
    st.session_state.df_results_av = pd.DataFrame()
    with st.spinner("Analyse Alpha Vantage en cours... (peut prendre plusieurs minutes à cause des limites API)"):
        df_temp_av = analyze_forex_pairs_alphavantage()
    
    if df_temp_av is not None and not df_temp_av.empty:
        st.session_state.df_results_av = df_temp_av
        st.session_state.data_loaded_av = True
        st.write("DEBUG Résultats Alpha Vantage stockés.")
    else:
        st.warning("Aucun résultat concluant (Alpha Vantage).")

if st.session_state.data_loaded_av and not st.session_state.df_results_av.empty:
    st.write("DEBUG Affichage des résultats Alpha Vantage.")
    df_to_display_av = st.session_state.df_results_av
    
    st.subheader("Classement des paires Forex (H1, D, W)")
    trend_columns_to_style = ['H1', 'D', 'W'] # H4 a été enlevé
    def style_trends(val):
        if val == 'Bullish': return 'background-color: mediumseagreen; color: white;'
        elif val == 'Bearish': return 'background-color: indianred; color: white;'
        elif val == 'Neutral': return 'background-color: khaki; color: black;'
        else: return 'background-color: whitesmoke; color: black;'
    
    # S'assurer que les colonnes à styler existent dans le DataFrame
    existing_style_cols = [col for col in trend_columns_to_style if col in df_to_display_av.columns]
    if existing_style_cols:
        styled_df_av = df_to_display_av.style.applymap(style_trends, subset=pd.IndexSlice[:, existing_style_cols])
        st.dataframe(styled_df_av, use_container_width=True)
    else:
        st.dataframe(df_to_display_av, use_container_width=True) # Afficher sans style si les colonnes manquent

    st.subheader("Résumé des Indicateurs")
    st.markdown("""
    - **H1 (1 Heure)**: Tendance basée sur HMA(12) vs EMA(20).
    - **D (Journalier)**: Tendance basée sur EMA(20) vs EMA(50).
    - **W (Hebdomadaire)**: Tendance basée sur EMA(20) vs EMA(50).
    - **H4**: Non disponible avec Alpha Vantage pour cette analyse.
    - **Score**: Somme des tendances (+1 Bullish, -1 Bearish, 0 Neutral) sur H1, D, W.
    """)
elif bouton_clique_av: # Si le bouton a été cliqué mais pas de données
    pass # Le warning est déjà affiché plus haut
else: # Si le bouton n'a pas encore été cliqué
    st.info("Cliquez sur le bouton pour lancer l'analyse Alpha Vantage.")
