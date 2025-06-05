# TOUT EN HAUT DU FICHIER APP.PY
import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

# Essayer d'importer TDClient, gérer l'erreur si l'import échoue
try:
    from twelvedata import TDClient
except ImportError as e:
    # Ce message s'affichera si la bibliothèque n'est pas installée,
    # mais st.set_page_config doit venir avant st.error
    # Donc on va stocker l'erreur et l'afficher après st.set_page_config
    TDCLIENT_IMPORT_ERROR = f"ERREUR D'IMPORT: Impossible d'importer TDClient: {e}. Assurez-vous que 'twelvedata' est dans requirements.txt et installé."
else:
    TDCLIENT_IMPORT_ERROR = None


# PREMIÈRE COMMANDE STREAMLIT
st.set_page_config(layout="wide")

# Afficher l'erreur d'import de TDClient ici si elle a eu lieu
if TDCLIENT_IMPORT_ERROR:
    st.error(TDCLIENT_IMPORT_ERROR)
    st.stop() # Arrêter l'application si l'import crucial échoue

st.write("Début du script - Imports OK, Page Config OK")


# Configuration Twelve Data (depuis st.secrets)
st.write("Tentative de lecture des secrets...")
try:
    TWELVEDATA_API_KEY = st.secrets["TWELVEDATA_API_KEY"]
    st.write("Secret TWELVEDATA_API_KEY lu avec succès.")
except KeyError:
    st.error("Erreur: Le secret TWELVEDATA_API_KEY n'est pas configuré dans st.secrets.")
    st.write("Veuillez configurer le secret TWELVEDATA_API_KEY dans .streamlit/secrets.toml (local) ou dans les paramètres de l'application (Streamlit Cloud).")
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

# Liste des paires Forex à analyser (format Twelve Data : EUR/USD)
forex_pairs_twelvedata = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD']

st.write("Définitions des fonctions et listes de paires OK.")

# --- Fonctions de calcul et d'analyse ---
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

def get_twelvedata_data(symbol, interval, output_size):
    st.write(f"DEBUG get_twelvedata_data: Appel pour {symbol}, interval={interval}, output_size={output_size}")
    try:
        ts = td.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=output_size,
            timezone="Exchange"
        )
        st.write(f"DEBUG get_twelvedata_data: Appel API td.time_series pour {symbol} ({interval}) effectué.")
        df = ts.as_pandas()
        st.write(f"DEBUG get_twelvedata_data: Conversion en Pandas pour {symbol} ({interval}) : {'Vide' if df is None or df.empty else 'OK, taille ' + str(len(df))}")

        if df is None or df.empty:
            st.warning(f"Aucune donnée Twelve Data retournée pour {symbol} ({interval})")
            return pd.DataFrame()

        column_mapping = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
        df.rename(columns=column_mapping, inplace=True, errors='ignore') # errors='ignore' au cas où une colonne n'existerait pas
        st.write(f"DEBUG get_twelvedata_data: Colonnes OHLC renommées pour {symbol} ({interval}). Colonnes actuelles: {df.columns.tolist()}")

        final_columns = ['Open', 'High', 'Low', 'Close']
        # Gérer la colonne Volume
        if 'volume' in df.columns: # Nom original de Twelve Data
            df.rename(columns={'volume': 'Volume'}, inplace=True)
            final_columns.append('Volume')
            st.write(f"DEBUG get_twelvedata_data: Colonne 'volume' renommée en 'Volume' pour {symbol} ({interval}).")
        elif 'Volume' in df.columns: # Si elle était déjà en majuscule
             final_columns.append('Volume')
             st.write(f"DEBUG get_twelvedata_data: Colonne 'Volume' (déjà majuscule) trouvée pour {symbol} ({interval}).")
        else:
            st.write(f"DEBUG get_twelvedata_data: Pas de colonne de volume ('volume' ou 'Volume') trouvée pour {symbol} ({interval}).")

        df.sort_index(ascending=True, inplace=True)

        required_ohlc = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_ohlc):
            st.error(f"Colonnes OHLC essentielles manquantes après traitement pour {symbol} ({interval}). Colonnes disponibles: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # S'assurer que seules les colonnes finales demandées sont retournées et dans le bon ordre
        # Et seulement si elles existent dans le DataFrame df
        existing_final_columns = [col for col in final_columns if col in df.columns]
        st.write(f"DEBUG get_twelvedata_data: Retour pour {symbol} ({interval}) avec {len(df)} lignes et colonnes {existing_final_columns}.")
        return df[existing_final_columns]

    except Exception as e:
        st.warning(f"Erreur DANS get_twelvedata_data pour {symbol} ({interval}): {e}")
        import traceback
        st.warning(traceback.format_exc())
        return pd.DataFrame()

def analyze_forex_pairs_twelvedata():
    st.write("DEBUG analyze_forex_pairs_twelvedata: Début de l'analyse.")
    results = []
    timeframe_params = {
        'H1': {'interval': '1h',    'outputsize': 250},
        'H4': {'interval': '4h',    'outputsize': 250},
        'D':  {'interval': '1day',  'outputsize': 200},
        'W':  {'interval': '1week', 'outputsize': 100}
    }

    for pair_symbol in forex_pairs_twelvedata:
        st.write(f"DEBUG analyze_forex_pairs_twelvedata: Traitement de {pair_symbol}.")
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
                if df.empty or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    st.error(f"analyze_forex_pairs_twelvedata: Données invalides ou OHLC manquantes pour {pair_symbol} ({tf_key}).")
                    all_data_fetched_successfully = False
                    break 
                data_sets[tf_key] = df
            
            if not all_data_fetched_successfully:
                st.write(f"DEBUG analyze_forex_pairs_twelvedata: Saut de {pair_symbol} car les données n'ont pas pu être toutes récupérées correctement.")
                continue

            data_1h, data_4h, data_d, data_w = data_sets['H1'], data_sets['H4'], data_sets['D'], data_sets['W']
            st.write(f"DEBUG analyze_forex_pairs_twelvedata: Calcul des indicateurs pour {pair_symbol}.")

            # Vérifier si la colonne 'Close' existe avant de calculer les indicateurs
            if 'Close' not in data_1h or 'Close' not in data_4h or \
               'Close' not in data_d or 'Close' not in data_w:
                st.error(f"analyze_forex_pairs_twelvedata: Colonne 'Close' manquante dans un des DataFrames pour {pair_symbol} avant calcul indicateurs.")
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
            continue

    st.write(f"DEBUG analyze_forex_pairs_twelvedata: Fin de l'analyse, {len(results)} résultats trouvés.")
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    return df

# --- Interface Streamlit ---
st.title("Classement des Paires Forex par Tendance MTF (via Twelve Data)")
st.write("Interface Streamlit initialisée (Titre OK).")

# Initialiser session_state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_results' not in st.session_state:
    st.session_state.df_results = pd.DataFrame() # Initialiser comme DataFrame vide

bouton_clique = st.button("Actualiser et Analyser les Paires Forex (Twelve Data)")
st.write(f"DEBUG État du bouton avant le if: {bouton_clique}")

if bouton_clique:
    st.write("DEBUG Bouton cliqué - DANS LE IF.")
    st.session_state.data_loaded = False # Reset flag pour forcer le rechargement/réaffichage
    st.session_state.df_results = pd.DataFrame() # Vider les anciens résultats

    # Utiliser le spinner ici, car l'analyse peut prendre du temps
    with st.spinner("Analyse des données Twelve Data en cours... Merci de patienter."):
        st.write("DEBUG Spinner activé, appel de analyze_forex_pairs_twelvedata...")
        df_results_temp = analyze_forex_pairs_twelvedata()
    
    st.write(f"DEBUG analyze_forex_pairs_twelvedata a retourné : {type(df_results_temp)}")
    if df_results_temp is not None and not df_results_temp.empty:
        st.session_state.df_results = df_results_temp
        st.session_state.data_loaded = True
        st.write("DEBUG Résultats stockés dans session_state.")
    else:
        st.warning("Aucun résultat concluant retourné par l'analyse ou DataFrame vide.")
        st.session_state.data_loaded = False # Assurer que c'est bien False


# Affichage conditionnel basé sur st.session_state
st.write(f"DEBUG État de data_loaded avant l'affichage des résultats: {st.session_state.data_loaded}")
st.write(f"DEBUG Type de df_results dans session_state: {type(st.session_state.df_results)}, Est-il vide: {st.session_state.df_results.empty if isinstance(st.session_state.df_results, pd.DataFrame) else 'N/A'}")


if st.session_state.data_loaded and isinstance(st.session_state.df_results, pd.DataFrame) and not st.session_state.df_results.empty:
    st.write("DEBUG Affichage des résultats depuis session_state...")
    df_to_display = st.session_state.df_results # Pas besoin de .get() ici car on a vérifié
    
    st.subheader("Classement des paires Forex")
    def style_trends(val):
        if val == 'Bullish': return 'background-color: mediumseagreen; color: white;'
        elif val == 'Bearish': return 'background-color: indianred; color: white;'
        elif val == 'Neutral': return 'background-color: khaki; color: black;'
        else: return 'background-color: whitesmoke; color: black;'
    styled_df = df_to_display.style.applymap(style_trends, subset=pd.IndexSlice[:, ['H1', 'H4', 'D', 'W']])
    st.dataframe(styled_df, use_container_width=True)

    st.subheader("Résumé des Indicateurs")
    st.markdown("""
    - **H1 (1 Heure)**: Tendance basée sur HMA(12) vs EMA(20).
    - **H4 (4 Heures)**: Tendance basée sur HMA(12) vs EMA(20).
    - **D (Journalier)**: Tendance basée sur EMA(20) vs EMA(50).
    - **W (Hebdomadaire)**: Tendance basée sur EMA(20) vs EMA(50).
    - **Score**: Somme des tendances (+1 pour Bullish, -1 pour Bearish, 0 pour Neutral) sur les 4 timeframes.
    """)
elif bouton_clique and (not st.session_state.data_loaded or (isinstance(st.session_state.df_results, pd.DataFrame) and st.session_state.df_results.empty)):
    # Ce cas est pour quand le bouton a été cliqué, l'analyse a tourné, mais n'a rien produit de valable.
    # Le message st.warning("Aucun résultat concluant...") devrait déjà s'être affiché.
    # On peut ajouter un message plus persistant si besoin.
    st.write("DEBUG: Bouton cliqué, mais pas de données valides à afficher.")
    pass # Le warning est déjà affiché plus haut.
elif not bouton_clique:
    st.info("Cliquez sur le bouton ci-dessus pour lancer l'analyse avec les données Twelve Data.")
