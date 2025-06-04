import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta, date # Ajout de date
from polygon import RESTClient # Client Polygon.io

# Configuration Polygon.io (depuis st.secrets)
try:
    POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]
except KeyError:
    st.error("Erreur: Le secret POLYGON_API_KEY n'est pas configuré.")
    st.stop()

# Initialisation du client Polygon.io
polygon_client = RESTClient(POLYGON_API_KEY)

# Liste des paires Forex à analyser (format Polygon.io : C:EURUSD ou X:EURUSD)
# Le client python semble préférer le format "X:TICKER" pour les devises
# ou "C:TICKER" pour les cryptos/devises. Testons avec "C:"
# Alternative : Ticker direct comme "EURUSD" pour polygon_client.get_aggs
forex_pairs_polygon = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
# Pour les devises, le ticker est souvent préfixé par "C:" ou "X:"
# Par exemple, pour EUR/USD, le ticker Forex est "C:EURUSD" ou "X:EURUSD".
# La fonction get_aggs accepte souvent le ticker simple comme "EURUSD" pour les devises si le marché est bien FX.
# Si vous rencontrez des problèmes, essayez de préfixer avec "X:" (ex: "X:EURUSD")


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


# Fonction pour récupérer les données Polygon.io et les formater
def get_polygon_data(ticker_symbol, timespan, multiplier, from_date, to_date, limit=5000):
    # Pour les devises, Polygon utilise souvent le préfixe "C:" ou "X:"
    # Ou parfois juste le ticker simple si le endpoint est spécifique au Forex.
    # La fonction `get_aggs` est générique. Tentons avec "C:"
    formatted_ticker = f"C:{ticker_symbol}"
    try:
        aggs = polygon_client.get_aggs(
            ticker=formatted_ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=from_date,
            to=to_date,
            adjusted=True,
            sort='asc',
            limit=limit # Max 5000 pour le plan gratuit, peut être 50000 pour payant
        )
        if not aggs:
            st.warning(f"Aucune donnée Polygon.io retournée pour {formatted_ticker} ({timespan}) de {from_date} à {to_date}")
            return pd.DataFrame()

        df = pd.DataFrame([{
            'time': pd.to_datetime(agg.timestamp, unit='ms'), # Timestamp est en millisecondes
            'Open': agg.open,
            'High': agg.high,
            'Low': agg.low,
            'Close': agg.close,
            'Volume': agg.volume
        } for agg in aggs])

        if df.empty:
            return pd.DataFrame()

        df = df.set_index('time')
        return df
    except Exception as e:
        st.warning(f"Erreur lors de la récupération des données Polygon.io pour {formatted_ticker} ({timespan}): {e}")
        # Essayer sans le préfixe "C:" si cela échoue (certains endpoints/clients le gèrent différemment)
        try:
            st.info(f"Nouvel essai pour {ticker_symbol} sans préfixe...")
            aggs_no_prefix = polygon_client.get_aggs(
                ticker=ticker_symbol, # Sans préfixe
                multiplier=multiplier,
                timespan=timespan,
                from_=from_date,
                to=to_date,
                adjusted=True,
                sort='asc',
                limit=limit
            )
            if not aggs_no_prefix:
                st.warning(f"Aucune donnée Polygon.io (sans préfixe) pour {ticker_symbol} ({timespan})")
                return pd.DataFrame()
            df_no_prefix = pd.DataFrame([{
                'time': pd.to_datetime(agg.timestamp, unit='ms'),
                'Open': agg.open, 'High': agg.high, 'Low': agg.low, 'Close': agg.close, 'Volume': agg.volume
            } for agg in aggs_no_prefix])
            if df_no_prefix.empty: return pd.DataFrame()
            df_no_prefix = df_no_prefix.set_index('time')
            return df_no_prefix
        except Exception as e2:
            st.error(f"Erreur persistante pour Polygon.io pour {ticker_symbol} ({timespan}): {e2}")
            return pd.DataFrame()


# Fonction pour analyser les paires Forex avec Polygon.io
def analyze_forex_pairs_polygon():
    results = []
    today = date.today() # Utiliser date pour Polygon

    # Définir les paramètres pour chaque timeframe pour Polygon
    # Polygon nécessite une plage de dates. Nous allons calculer la date de début.
    # HMA(12) ~15 pts, EMA(20) ~20, EMA(50) ~50.
    # Il faut suffisamment de jours/semaines/heures pour obtenir ces points.
    # Note: Le 'limit' par défaut est 5000, ce qui est beaucoup.
    #       Nous allons nous assurer que notre plage de dates ne génère pas plus.
    #       Mais pour la plupart des timeframes, la plage suffira.

    timeframe_params = {
        'H1': {'timespan': 'hour', 'multiplier': 1, 'days_back': 20}, # ~20 jours * 24h = 480 barres
        'H4': {'timespan': 'hour', 'multiplier': 4, 'days_back': 80}, # ~80 jours * 6 (4h-barres/j) = 480 barres
        'D':  {'timespan': 'day',  'multiplier': 1, 'days_back': 250},# ~250 barres
        'W':  {'timespan': 'week', 'multiplier': 1, 'days_back': 3*365} # ~3 ans * 52 sem = 156 barres
    }

    for pair_symbol in forex_pairs_polygon:
        try:
            data_sets = {}
            for tf_key, params in timeframe_params.items():
                from_d = today - timedelta(days=params['days_back'])
                # Formater les dates en string YYYY-MM-DD
                from_date_str = from_d.strftime('%Y-%m-%d')
                to_date_str = today.strftime('%Y-%m-%d')

                df = get_polygon_data(
                    ticker_symbol=pair_symbol,
                    timespan=params['timespan'],
                    multiplier=params['multiplier'],
                    from_date=from_date_str,
                    to_date=to_date_str
                )
                if df.empty:
                    st.error(f"Données Polygon.io manquantes pour {pair_symbol} ({tf_key})")
                    # Marquer cette paire comme invalide pour l'analyse
                    data_sets[tf_key] = pd.DataFrame() # Mettre un DF vide pour la vérification ci-dessous
                    break # Arrêter de traiter les timeframes pour cette paire
                data_sets[tf_key] = df
            
            # Si une des récupérations a échoué pour cette paire
            if any(df.empty for df in data_sets.values()) or len(data_sets) != len(timeframe_params):
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
                st.error(f"Valeurs manquantes critiques dans les indicateurs Polygon pour {pair_symbol}.")
                continue

            trend_1h = get_trend(hma12_1h, ema20_1h)
            trend_4h = get_trend(hma12_4h, ema20_4h)
            trend_d = get_trend(ema20_d, ema50_d)
            trend_w = get_trend(ema20_w, ema50_w)

            score = sum([1 if t == 'Bullish' else -1 if t == 'Bearish' else 0 for t in [trend_1h, trend_4h, trend_d, trend_w]])

            results.append({
                'Pair': f"{pair_symbol[:3]}/{pair_symbol[3:]}", # ex: EURUSD -> EUR/USD
                'H1': trend_1h,
                'H4': trend_4h,
                'D': trend_d,
                'W': trend_w,
                'Score': score
            })
        except Exception as e:
            st.error(f"Erreur générale pour la paire Polygon {pair_symbol}: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            continue

    if not results:
        st.error("Aucune donnée Polygon.io valide n'a pu être récupérée ou analysée.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    return df

# --- Interface Streamlit (reste presque identique) ---
st.set_page_config(layout="wide")
st.title("Classement des Paires Forex par Tendance MTF (via Polygon.io)")

if st.button("Actualiser et Analyser les Paires Forex (Polygon.io)"):
    with st.spinner("Analyse des données Polygon.io en cours... Merci de patienter."):
        df_results = analyze_forex_pairs_polygon() # Appel de la nouvelle fonction

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
    st.info("Cliquez sur le bouton ci-dessus pour lancer l'analyse avec les données Polygon.io.")
