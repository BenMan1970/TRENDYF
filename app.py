import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from twelvedata import TDClient # Client Twelve Data

# Configuration Twelve Data (depuis st.secrets)
try:
    TWELVEDATA_API_KEY = st.secrets["TWELVEDATA_API_KEY"]
except KeyError:
    st.error("Erreur: Le secret TWELVEDATA_API_KEY n'est pas configuré.")
    st.stop()

# Initialisation du client Twelve Data
td = TDClient(apikey=TWELVEDATA_API_KEY)

# Liste des paires Forex à analyser (format Twelve Data : EUR/USD)
forex_pairs_twelvedata = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD']

# --- Les fonctions hma et get_trend restent inchangées ---
def hma(series, length):
    length = int(length)
    min_points_needed = length + int(math.sqrt(length)) -1
    if len(series) < min_points_needed:
        return pd.Series([np.nan] * len(series), index=series.index, name='HMA')
    wma1_period = int(length / 2)
    sqrt_length_period = int(math.sqrt(length))
    if wma1_period < 1 or length < 1 or sqrt_length_period < 1:
        return pd.Series([np.nan] * len(series), index=series.index, name='HMA')
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
# --- Fin des fonctions inchangées ---


# Fonction pour récupérer les données Twelve Data et les formater
def get_twelvedata_data(symbol, interval, output_size):
    try:
        # Utilisation de la méthode TimeSeries
        ts = td.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=output_size,
            timezone="Exchange" # ou "UTC"
        )
        # La bibliothèque peut retourner les données dans différents formats.
        # as_pandas() est pratique.
        df = ts.as_pandas()

        if df is None or df.empty:
            st.warning(f"Aucune donnée Twelve Data retournée pour {symbol} ({interval})")
            return pd.DataFrame()

        # Renommer les colonnes pour correspondre à ce que le reste du code attend
        # Twelve Data retourne 'open', 'high', 'low', 'close', 'volume'
        # L'index est déjà un DatetimeIndex
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        # S'assurer que l'index est trié par ordre croissant (le plus ancien en premier)
        # La bibliothèque devrait le faire, mais une vérification ne fait pas de mal.
        df.sort_index(ascending=True, inplace=True)

        return df[['Open', 'High', 'Low', 'Close', 'Volume']] # Sélectionner les colonnes nécessaires

    except Exception as e:
        st.warning(f"Erreur lors de la récupération des données Twelve Data pour {symbol} ({interval}): {e}")
        return pd.DataFrame()


# Fonction pour analyser les paires Forex avec Twelve Data
def analyze_forex_pairs_twelvedata():
    results = []

    # Définir les paramètres pour chaque timeframe pour Twelve Data
    # HMA(12) ~15 pts, EMA(20) ~20, EMA(50) ~50.
    # `outputsize` est le nombre de points de données à récupérer.
    timeframe_params = {
        'H1': {'interval': '1h',    'outputsize': 250}, # Assez pour HMA(12), EMA(20)
        'H4': {'interval': '4h',    'outputsize': 250}, # Assez pour HMA(12), EMA(20)
        'D':  {'interval': '1day',  'outputsize': 200}, # Assez pour EMA(20), EMA(50)
        'W':  {'interval': '1week', 'outputsize': 100}  # Assez pour EMA(20), EMA(50)
    }

    for pair_symbol in forex_pairs_twelvedata:
        try:
            data_sets = {}
            all_data_fetched = True
            for tf_key, params in timeframe_params.items():
                df = get_twelvedata_data(
                    symbol=pair_symbol,
                    interval=params['interval'],
                    output_size=params['outputsize']
                )
                if df.empty:
                    st.error(f"Données Twelve Data manquantes pour {pair_symbol} ({tf_key})")
                    all_data_fetched = False
                    break # Arrêter de traiter les timeframes pour cette paire
                data_sets[tf_key] = df
            
            if not all_data_fetched:
                continue # Passer à la paire suivante

            data_1h, data_4h, data_d, data_w = data_sets['H1'], data_sets['H4'], data_sets['D'], data_sets['W']

            # Les calculs d'indicateurs et la logique de tendance restent les mêmes
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
                st.error(f"Valeurs manquantes critiques dans les indicateurs Twelve Data pour {pair_symbol}.")
                continue

            trend_1h = get_trend(hma12_1h, ema20_1h)
            trend_4h = get_trend(hma12_4h, ema20_4h)
            trend_d = get_trend(ema20_d, ema50_d)
            trend_w = get_trend(ema20_w, ema50_w)

            score = sum([1 if t == 'Bullish' else -1 if t == 'Bearish' else 0 for t in [trend_1h, trend_4h, trend_d, trend_w]])

            results.append({
                'Pair': pair_symbol, # Déjà au format EUR/USD
                'H1': trend_1h,
                'H4': trend_4h,
                'D': trend_d,
                'W': trend_w,
                'Score': score
            })
        except Exception as e:
            st.error(f"Erreur générale pour la paire Twelve Data {pair_symbol}: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            continue

    if not results:
        st.error("Aucune donnée Twelve Data valide n'a pu être récupérée ou analysée.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    return df

# --- Interface Streamlit (reste presque identique) ---
st.set_page_config(layout="wide")
st.title("Classement des Paires Forex par Tendance MTF (via Twelve Data)")

if st.button("Actualiser et Analyser les Paires Forex (Twelve Data)"):
    with st.spinner("Analyse des données Twelve Data en cours... Merci de patienter."):
        df_results = analyze_forex_pairs_twelvedata() # Appel de la nouvelle fonction

        if not df_results.empty:
            st.subheader("Classement des paires Forex")
            def style_trends(val):
                if val == 'Bullish': return 'background-color: mediumseagreen; color: white;'
                elif val == 'Bearish': return 'background-color: indianred; color: white;'
                elif val == 'Neutral': return 'background-color: khaki; color: black;'
                else: return 'background-color: whitesmoke; color: black;'
            styled_df = df_results.style.applymap(style_trends, subset=['H1', 'H4', 'D', 'W'])
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
            st.warning("Aucun résultat à afficher. Essayez de réactualiser ou vérifiez les logs d'erreur.")
else:
    st.info("Cliquez sur le bouton ci-dessus pour lancer l'analyse avec les données Twelve Data.")
