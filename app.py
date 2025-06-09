import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import requests
import time

# --- Configuration de la Page (doit être la première commande Streamlit) ---
st.set_page_config(layout="wide")

# --- Lecture des Secrets et Initialisation API ---
FMP_API_KEY = None
try:
    FMP_API_KEY = st.secrets["FMP_API_KEY"]
except KeyError:
    st.error("Erreur Critique: Le secret FMP_API_KEY n'est pas configuré. L'application ne peut pas fonctionner.")
    st.stop()
except Exception as e:
    st.error(f"Erreur Critique lors de la lecture des secrets FMP: {e}. L'application ne peut pas fonctionner.")
    st.stop()

# --- Constantes ---

# MODIFIÉ: Liste étendue de paires Forex (Majeures et Mineures/Croisées)
FOREX_PAIRS_EXTENDED = [
    # Majeures
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
    
    # Mineures (Croisées de l'Euro)
    'EURGBP', 'EURJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
    
    # Mineures (Croisées de la Livre Sterling)
    'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD',
    
    # Mineures (Croisées du Dollar Australien)
    'AUDJPY', 'AUDCAD', 'AUDCHF', 'AUDNZD',
    
    # Mineures (Croisées du Dollar Canadien)
    'CADJPY', 'CADCHF',
    
    # Mineures (Croisées du Franc Suisse)
    'CHFJPY',
    
    # Mineures (Croisées du Dollar Néo-Zélandais)
    'NZDJPY', 'NZDCAD', 'NZDCHF'
]

# --- Fonctions Indicateurs et Tendance (inchangées) ---
def hma(series, length):
    length = int(length)
    min_points_needed = length + int(math.sqrt(length)) - 1
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

# --- Fonction de Récupération de Données FMP (inchangée et parfaitement adaptée) ---
def get_fmp_data(symbol, interval_fmp, from_date_str=None, to_date_str=None):
    base_url = "https://financialmodelingprep.com/api/v3/"
    params = {'apikey': FMP_API_KEY}
    data_key_in_json = None

    if interval_fmp in ['1hour', '4hour']:
        endpoint = f"historical-chart/{interval_fmp}/{symbol}"
    elif interval_fmp == '1day':
        endpoint = f"historical-price-full/{symbol}"
        data_key_in_json = "historical"
        if from_date_str: params['from'] = from_date_str
        if to_date_str: params['to'] = to_date_str
    else:
        print(f"Erreur interne: Intervalle FMP non géré: {interval_fmp}") 
        return pd.DataFrame()

    full_url = base_url + endpoint
    try:
        response = requests.get(full_url, params=params, timeout=30)
        response.raise_for_status()
        raw_data = response.json()

        if not raw_data or (isinstance(raw_data, dict) and raw_data.get('Error Message')):
            error_msg = raw_data.get('Error Message', 'Réponse vide/invalide de FMP.') if isinstance(raw_data, dict) else 'Réponse vide de FMP.'
            st.toast(f"API FMP Info pour {symbol} ({interval_fmp}): {error_msg}", icon="⚠️")
            return pd.DataFrame()

        data_list = raw_data[data_key_in_json] if data_key_in_json and data_key_in_json in raw_data else raw_data
        if not isinstance(data_list, list) or not data_list :
             st.toast(f"Format de données inattendu ou vide pour {symbol} ({interval_fmp}).", icon="⚠️")
             return pd.DataFrame()

        df = pd.DataFrame(data_list)
        if df.empty: return pd.DataFrame()
        if 'date' not in df.columns: return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        df.rename(columns=rename_map, inplace=True)
        df.sort_index(ascending=True, inplace=True)

        final_cols = ['Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns: final_cols.append('Volume')
        existing_final_cols = [c for c in final_cols if c in df.columns]
        
        if not all(c in existing_final_cols for c in ['Open', 'High', 'Low', 'Close']):
            return pd.DataFrame()
        return df[existing_final_cols]

    except requests.exceptions.Timeout:
        st.toast(f"Timeout API pour {symbol} ({interval_fmp})", icon="⏱️")
        return pd.DataFrame()
    except requests.exceptions.HTTPError as http_err:
        st.toast(f"Erreur HTTP FMP ({http_err.response.status_code}) pour {symbol} ({interval_fmp})", icon="🔥")
        return pd.DataFrame()
    except Exception:
        st.toast(f"Erreur de traitement des données pour {symbol} ({interval_fmp})", icon="⚙️")
        import traceback
        print(f"Erreur détaillée dans get_fmp_data pour {symbol}, {interval_fmp}:\n{traceback.format_exc()}")
        return pd.DataFrame()

# --- Fonction d'Analyse Principale (modifiée pour la nouvelle liste) ---
# DÉCOMMENTEZ LA LIGNE CI-DESSOUS pour activer la mise en cache, c'est recommandé !
# @st.cache_data(ttl=60*5) 
def analyze_forex_pairs():
    results_internal = []
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    timeframe_params_fmp = {
        'H1': {'interval_api': '1hour', 'days_history_for_daily_weekly': None},
        'H4': {'interval_api': '4hour', 'days_history_for_daily_weekly': None},
        'D':  {'interval_api': '1day',  'days_history_for_daily_weekly': 365 * 1},
        'W':  {'interval_api': '1day',  'days_history_for_daily_weekly': 365 * 3}
    }
    
    total_pairs = len(FOREX_PAIRS_EXTENDED)
    progress_bar = st.progress(0, text=f"Analyse de 0 / {total_pairs} paires...")
    processed_pairs_count = 0

    # MODIFIÉ: Boucle sur la nouvelle liste étendue de paires Forex
    for i, pair_symbol in enumerate(FOREX_PAIRS_EXTENDED):
        try:
            data_sets = {}
            all_data_ok = True
            for tf_key, params in timeframe_params_fmp.items():
                from_date_req_str = None
                if params['days_history_for_daily_weekly']:
                    from_date_req = today - timedelta(days=params['days_history_for_daily_weekly'])
                    from_date_req_str = from_date_req.strftime('%Y-%m-%d')
                
                df = get_fmp_data(
                    symbol=pair_symbol,
                    interval_fmp=params['interval_api'],
                    from_date_str=from_date_req_str,
                    to_date_str=today_str if from_date_req_str else None
                )
                
                if tf_key == 'W' and not df.empty:
                    agg_funcs = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
                    if 'Volume' in df.columns: agg_funcs['Volume'] = 'sum'
                    df_weekly = df.resample('W-FRI').agg(agg_funcs)
                    df = df_weekly.dropna()

                if df.empty or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    all_data_ok = False; break
                
                data_sets[tf_key] = df

            if not all_data_ok:
                st.toast(f"Analyse de {pair_symbol} sautée (données manquantes).", icon="ℹ️"); continue

            data_h1, data_h4, data_d, data_w = data_sets['H1'], data_sets['H4'], data_sets['D'], data_sets['W']
            hma12_h1 = hma(data_h1['Close'], 12); ema20_h1 = data_h1['Close'].ewm(span=20, adjust=False).mean()
            hma12_h4 = hma(data_h4['Close'], 12); ema20_h4 = data_h4['Close'].ewm(span=20, adjust=False).mean()
            ema20_d  = data_d['Close'].ewm(span=20, adjust=False).mean(); ema50_d = data_d['Close'].ewm(span=50, adjust=False).mean()
            ema20_w  = data_w['Close'].ewm(span=20, adjust=False).mean(); ema50_w = data_w['Close'].ewm(span=50, adjust=False).mean()

            indicators = [hma12_h1, ema20_h1, hma12_h4, ema20_h4, ema20_d, ema50_d, ema20_w, ema50_w]
            if any(s is None or s.dropna().empty for s in indicators):
                st.toast(f"Calcul indicateurs incomplet pour {pair_symbol}, saut.", icon="🟠"); continue

            trend_h1 = get_trend(hma12_h1, ema20_h1); trend_h4 = get_trend(hma12_h4, ema20_h4)
            trend_d  = get_trend(ema20_d, ema50_d); trend_w  = get_trend(ema20_w, ema50_w)
            score = sum([1 if t == 'Bullish' else -1 if t == 'Bearish' else 0 for t in [trend_h1, trend_h4, trend_d, trend_w]])
            
            # MODIFIÉ: Le nom de la colonne redevient "Paire" et utilise le formatage simple
            results_internal.append({'Paire': f"{pair_symbol[:3]}/{pair_symbol[3:]}", 
                                     'H1': trend_h1, 'H4': trend_h4, 'D': trend_d, 'W': trend_w, 
                                     '_score_internal': score})
            processed_pairs_count += 1
        
        except Exception as e:
            st.toast(f"Erreur d'analyse pour {pair_symbol}, saut.", icon="⚙️")
            import traceback
            print(f"Erreur détaillée dans analyze_forex_pairs pour {pair_symbol}:\n{traceback.format_exc()}")
            continue
        finally:
             progress_bar.progress((i + 1) / total_pairs, text=f"Analyse de {i+1} / {total_pairs} paires...")

    progress_bar.empty()

    if not results_internal:
        st.warning("Aucune donnée n'a pu être analysée avec succès.")
        return pd.DataFrame()

    if processed_pairs_count < total_pairs:
        st.info(f"{processed_pairs_count} sur {total_pairs} paires traitées. Certaines données pourraient manquer.")
    
    df_temp = pd.DataFrame(results_internal)
    df_temp = df_temp.sort_values(by='_score_internal', ascending=False).reset_index(drop=True)
    
    df_to_return = df_temp[['Paire', 'H1', 'H4', 'D', 'W']]
    return df_to_return

# --- Interface Utilisateur Streamlit ---
st.title("Classement des Paires Forex par Tendance MTF")

if 'df_results' not in st.session_state: st.session_state.df_results = pd.DataFrame()
if 'analysis_done_once' not in st.session_state: st.session_state.analysis_done_once = False

if st.button("🚀 Analyser les Paires Forex"):
    with st.spinner("Analyse des paires en cours... Cela peut prendre un moment."):
        # MODIFIÉ: Appel de la fonction et variable de session mises à jour
        st.session_state.df_results = analyze_forex_pairs()
        st.session_state.analysis_done_once = True

if st.session_state.analysis_done_once:
    if not st.session_state.df_results.empty:
        st.subheader("Classement des paires Forex")
        df_to_display = st.session_state.df_results
        
        def style_trends(val):
            if val == 'Bullish': return 'background-color: #2E7D32; color: white;'
            elif val == 'Bearish': return 'background-color: #C62828; color: white;'
            elif val == 'Neutral': return 'background-color: #FFD700; color: black;'
            else: return ''
        
        # Le code de stylisation s'adapte automatiquement
        styled_df = df_to_display.style.applymap(style_trends, subset=['H1', 'H4', 'D', 'W'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

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


