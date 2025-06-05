import streamlit as st
st.write("Début du script - Import Streamlit OK") # PREMIER LOG

import pandas as pd
st.write("Import Pandas OK")
import numpy as np
st.write("Import Numpy OK")
import math
st.write("Import Math OK")
from datetime import datetime, timedelta
st.write("Import Datetime OK")

try:
    from twelvedata import TDClient
    st.write("Import TDClient OK")
except ImportError as e:
    st.error(f"ERREUR D'IMPORT: Impossible d'importer TDClient: {e}")
    st.stop()


# Configuration Twelve Data (depuis st.secrets)
st.write("Tentative de lecture des secrets...")
try:
    TWELVEDATA_API_KEY = st.secrets["TWELVEDATA_API_KEY"]
    st.write("Secret TWELVEDATA_API_KEY lu avec succès.")
except KeyError:
    st.error("Erreur: Le secret TWELVEDATA_API_KEY n'est pas configuré dans st.secrets.")
    st.write("Veuillez configurer le secret TWELVEDATA_API_KEY.")
    st.stop()
except Exception as e:
    st.error(f"Erreur inattendue lors de la lecture des secrets: {e}")
    st.stop()

# Initialisation du client Twelve Data
st.write("Tentative d'initialisation du client TDClient...")
try:
    td = TDClient(apikey=TWELVEDATA_API_KEY)
    st.write("Client TDClient initialisé avec succès.")
except Exception as e:
    st.error(f"Erreur lors de l'initialisation de TDClient: {e}")
    import traceback
    st.error(traceback.format_exc())
    st.stop()

# --- Le reste de votre code (listes de paires, fonctions hma, get_trend, etc.) ---
# ... (gardez vos fonctions telles quelles pour l'instant) ...

st.write("Définitions des fonctions et listes OK.") # LOG AVANT L'INTERFACE

# Interface Streamlit
st.set_page_config(layout="wide")
st.title("Classement des Paires Forex par Tendance MTF (via Twelve Data)")
st.write("Interface Streamlit initialisée.") # LOG APRES LE TITRE

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if st.button("Actualiser et Analyser les Paires Forex (Twelve Data)"):
    st.write("Bouton cliqué.") # LOG CLIC BOUTON
    st.session_state.data_loaded = False # Reset flag
    with st.spinner("Analyse des données Twelve Data en cours... Merci de patienter."):
        st.write("Spinner activé, appel de analyze_forex_pairs_twelvedata...")
        df_results = analyze_forex_pairs_twelvedata()
        st.write(f"analyze_forex_pairs_twelvedata a retourné : {type(df_results)}")
        if df_results is not None and not df_results.empty:
            st.session_state.df_results = df_results
            st.session_state.data_loaded = True
            st.write("Résultats stockés dans session_state.")
        else:
            st.warning("Aucun résultat retourné par l'analyse.")
            st.session_state.data_loaded = False # Assurer que c'est bien False


if st.session_state.data_loaded:
    st.write("Affichage des résultats depuis session_state...")
    df_to_display = st.session_state.get('df_results')
    if df_to_display is not None and not df_to_display.empty:
        st.subheader("Classement des paires Forex")
        def style_trends(val):
            if val == 'Bullish': return 'background-color: mediumseagreen; color: white;'
            elif val == 'Bearish': return 'background-color: indianred; color: white;'
            elif val == 'Neutral': return 'background-color: khaki; color: black;'
            else: return 'background-color: whitesmoke; color: black;'
        styled_df = df_to_display.style.applymap(style_trends, subset=['H1', 'H4', 'D', 'W'])
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
        st.warning("Les données étaient attendues mais n'ont pas pu être affichées (vides ou None).")

elif not st.session_state.get('data_loaded', False) and 'df_results' not in st.session_state : # Si le bouton n'a pas encore été cliqué ou si l'analyse a échoué sans remplir data_loaded
    st.info("Cliquez sur le bouton ci-dessus pour lancer l'analyse avec les données Twelve Data.")

# Remettez vos fonctions analyze_forex_pairs_twelvedata, get_twelvedata_data, hma, get_trend ici.
# ... (votre code des fonctions ici) ...
# Fonction pour récupérer les données Twelve Data et les formater
def get_twelvedata_data(symbol, interval, output_size):
    st.write(f"Appel get_twelvedata_data pour {symbol}, {interval}, {output_size}")
    try:
        ts = td.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=output_size,
            timezone="Exchange" 
        )
        st.write(f"Appel API td.time_series pour {symbol} ({interval}) effectué.")
        df = ts.as_pandas()
        st.write(f"Conversion en Pandas pour {symbol} ({interval}) : {'Vide' if df is None or df.empty else 'OK, taille ' + str(len(df))}")


        if df is None or df.empty:
            st.warning(f"Aucune donnée Twelve Data retournée pour {symbol} ({interval})")
            return pd.DataFrame()

        column_mapping = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
        df.rename(columns=column_mapping, inplace=True)
        st.write(f"Colonnes renommées pour {symbol} ({interval}).")


        final_columns = ['Open', 'High', 'Low', 'Close']
        if 'volume' in df.columns:
            df.rename(columns={'volume': 'Volume'}, inplace=True)
            final_columns.append('Volume')
            st.write(f"Colonne Volume traitée pour {symbol} ({interval}).")
        elif 'Volume' in df.columns:
             final_columns.append('Volume')
             st.write(f"Colonne Volume (déjà majuscule) traitée pour {symbol} ({interval}).")
        else:
            st.write(f"Pas de colonne Volume trouvée pour {symbol} ({interval}).")


        df.sort_index(ascending=True, inplace=True)

        required_ohlc = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_ohlc):
            st.error(f"Colonnes OHLC manquantes après traitement pour {symbol} ({interval}). Colonnes: {df.columns}")
            return pd.DataFrame()
        
        st.write(f"Retour de get_twelvedata_data pour {symbol} ({interval}) avec {len(df)} lignes et colonnes {df.columns.tolist()}.")
        return df[final_columns]

    except Exception as e:
        st.warning(f"Erreur DANS get_twelvedata_data pour {symbol} ({interval}): {e}")
        import traceback
        st.warning(traceback.format_exc())
        return pd.DataFrame()

# ... (le reste de vos fonctions : analyze_forex_pairs_twelvedata, hma, get_trend) ...
# Assurez-vous que ces fonctions sont bien définies dans votre script.
# Je vais inclure analyze_forex_pairs_twelvedata ici pour la complétude des logs.

forex_pairs_twelvedata = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD']


def analyze_forex_pairs_twelvedata():
    st.write("analyze_forex_pairs_twelvedata: Début de l'analyse.")
    results = []
    timeframe_params = {
        'H1': {'interval': '1h',    'outputsize': 250},
        'H4': {'interval': '4h',    'outputsize': 250},
        'D':  {'interval': '1day',  'outputsize': 200},
        'W':  {'interval': '1week', 'outputsize': 100}
    }

    for pair_symbol in forex_pairs_twelvedata:
        st.write(f"analyze_forex_pairs_twelvedata: Traitement de {pair_symbol}.")
        try:
            data_sets = {}
            all_data_fetched = True
            for tf_key, params in timeframe_params.items():
                st.write(f"analyze_forex_pairs_twelvedata: Récupération {tf_key} pour {pair_symbol}.")
                df = get_twelvedata_data(
                    symbol=pair_symbol,
                    interval=params['interval'],
                    output_size=params['outputsize']
                )
                if df.empty:
                    st.error(f"analyze_forex_pairs_twelvedata: Données vides pour {pair_symbol} ({tf_key}).")
                    all_data_fetched = False
                    break 
                data_sets[tf_key] = df
            
            if not all_data_fetched:
                st.write(f"analyze_forex_pairs_twelvedata: Saut de {pair_symbol} car toutes les données n'ont pas été récupérées.")
                continue

            data_1h, data_4h, data_d, data_w = data_sets['H1'], data_sets['H4'], data_sets['D'], data_sets['W']
            st.write(f"analyze_forex_pairs_twelvedata: Calcul des indicateurs pour {pair_symbol}.")

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
                st.error(f"analyze_forex_pairs_twelvedata: Indicateurs avec NaN pour {pair_symbol}.")
                continue

            trend_1h = get_trend(hma12_1h, ema20_1h)
            trend_4h = get_trend(hma12_4h, ema20_4h)
            trend_d = get_trend(ema20_d, ema50_d)
            trend_w = get_trend(ema20_w, ema50_w)

            score = sum([1 if t == 'Bullish' else -1 if t == 'Bearish' else 0 for t in [trend_1h, trend_4h, trend_d, trend_w]])
            st.write(f"analyze_forex_pairs_twelvedata: Score {score} pour {pair_symbol}.")

            results.append({'Pair': pair_symbol, 'H1': trend_1h, 'H4': trend_4h, 'D': trend_d, 'W': trend_w, 'Score': score})
        except Exception as e:
            st.error(f"analyze_forex_pairs_twelvedata: Erreur générale pour {pair_symbol}: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            continue

    st.write(f"analyze_forex_pairs_twelvedata: Fin de l'analyse, {len(results)} résultats.")
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    return df

# Mettez ici les définitions des fonctions hma et get_trend telles que vous les aviez
def hma(series, length):
    # ... (code de hma) ...
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
    # ... (code de get_trend) ...
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
