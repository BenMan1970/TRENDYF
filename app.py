import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

# Liste des paires Forex à analyser
forex_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X', 'NZDUSD=X']

# Fonction HMA (Hull Moving Average) - utilise des SMA comme dans votre code
def hma(series, length):
    # S'assurer que la longueur est un entier
    length = int(length)
    # S'assurer qu'il y a assez de données pour les calculs de rolling window
    # Pour HMA(length), on a besoin d'au moins 'length' points pour la WMA interne
    # et 'sqrt(length)' points pour la WMA externe.
    # La première SMA(length) a besoin de 'length' points.
    # La première SMA(length/2) a besoin de 'length/2' points.
    # La SMA finale(sqrt(length)) a besoin de 'sqrt(length)' points sur raw_hma.
    # Total minimum de points nécessaires approx: length + int(math.sqrt(length)) -1
    min_points_needed = length + int(math.sqrt(length)) -1
    if len(series) < min_points_needed:
        # Retourner une série de NaN de la même taille si pas assez de données
        return pd.Series([np.nan] * len(series), index=series.index, name='HMA')

    wma1_period = int(length / 2)
    sqrt_length_period = int(math.sqrt(length))

    # Vérifier si les fenêtres sont valides (au moins 1)
    if wma1_period < 1 or length < 1 or sqrt_length_period < 1:
        return pd.Series([np.nan] * len(series), index=series.index, name='HMA')

    wma1 = series.rolling(window=wma1_period).mean() * 2
    wma2 = series.rolling(window=length).mean()
    
    # raw_hma peut avoir des NaNs au début à cause de wma2 (fenêtre plus longue)
    # Il faut s'assurer que la soustraction se fait correctement sur les index alignés
    raw_hma = wma1 - wma2 
    
    # La HMA finale
    hma_series = raw_hma.rolling(window=sqrt_length_period).mean()
    return hma_series

# Fonction pour déterminer la tendance
def get_trend(fast, slow):
    # Vérifier si les séries sont valides et contiennent des données
    if fast is None or slow is None or fast.empty or slow.empty:
        return 'Neutral'
    
    # Obtenir la dernière valeur non-NaN
    fast_last = fast.dropna().iloc[-1] if not fast.dropna().empty else np.nan
    slow_last = slow.dropna().iloc[-1] if not slow.dropna().empty else np.nan
    
    if pd.isna(fast_last) or pd.isna(slow_last):
        return 'Neutral'

    # Comparer les valeurs
    if fast_last > slow_last:
        return 'Bullish'
    elif fast_last < slow_last:
        return 'Bearish'
    else:
        return 'Neutral'

# Fonction pour récupérer et analyser les données
def analyze_forex_pairs():
    results = []
    
    for pair in forex_pairs:
        st.write(f"Analyse de {pair}...") # Feedback pour l'utilisateur
        try:
            # Télécharger les données pour différents timeframes
            # Augmenter les périodes pour assurer assez de données pour les indicateurs
            # HMA(12) a besoin d'environ 14 points. EMA(20) ~20 points. EMA(50) ~50 points.
            # Pour 1H: HMA(12), EMA(20) -> besoin d'au moins ~20 barres.
            #   '10d' (10 jours * 24h = 240 barres) devrait être largement suffisant.
            data_1h = yf.download(pair, period='10d', interval='1h', progress=False, auto_adjust=False, timeout=10)
            # Pour 4H: HMA(12), EMA(20) -> besoin d'au moins ~20 barres.
            #   '2mo' (2 mois * ~20 jours/mois * 6 barres/jour = 240 barres) devrait suffire.
            data_4h = yf.download(pair, period='2mo', interval='4h', progress=False, auto_adjust=False, timeout=10)
            # Pour D: EMA(20), EMA(50) -> besoin d'au moins ~50 barres.
            #   '8mo' (8 mois * ~20 jours/mois = 160 barres) devrait suffire.
            data_d = yf.download(pair, period='8mo', interval='1d', progress=False, auto_adjust=False, timeout=10)
            # Pour W: EMA(20), EMA(50) -> besoin d'au moins ~50 barres.
            #   '3y' (3 ans * 52 semaines = 156 barres) devrait suffire.
            data_w = yf.download(pair, period='3y', interval='1wk', progress=False, auto_adjust=False, timeout=10)

            # Vérifier si les données sont vides
            if data_1h.empty or data_4h.empty or data_d.empty or data_w.empty:
                st.error(f"Données manquantes pour la paire {pair} après téléchargement: H1={len(data_1h)}, H4={len(data_4h)}, D={len(data_d)}, W={len(data_w)}")
                continue

            st.info(f"Données téléchargées pour {pair}: H1={len(data_1h)}, H4={len(data_4h)}, D={len(data_d)}, W={len(data_w)}")

            # Calculer les moyennes mobiles
            hma12_1h = hma(data_1h['Close'], 12)
            ema20_1h = data_1h['Close'].ewm(span=20, adjust=False).mean()
            
            hma12_4h = hma(data_4h['Close'], 12)
            ema20_4h = data_4h['Close'].ewm(span=20, adjust=False).mean()
            
            ema20_d = data_d['Close'].ewm(span=20, adjust=False).mean()
            ema50_d = data_d['Close'].ewm(span=50, adjust=False).mean()
            
            ema20_w = data_w['Close'].ewm(span=20, adjust=False).mean()
            ema50_w = data_w['Close'].ewm(span=50, adjust=False).mean()

            # Vérifier si les calculs contiennent des NaN à la dernière valeur pertinente
            indicators = [hma12_1h, ema20_1h, hma12_4h, ema20_4h, ema20_d, ema50_d, ema20_w, ema50_w]
            
            # On vérifie la dernière valeur NON-NAN de chaque indicateur.
            # Si une série d'indicateur est entièrement NaN (ou vide après dropna), cela posera problème.
            
            valid_indicators = True
            for ind_series in indicators:
                if ind_series is None or ind_series.dropna().empty:
                    valid_indicators = False
                    break
            
            if not valid_indicators:
                st.error(f"Valeurs manquantes critiques dans les indicateurs pour {pair}. Un ou plusieurs indicateurs sont entièrement NaN.")
                continue

            # Déterminer les tendances
            trend_1h = get_trend(hma12_1h, ema20_1h)
            trend_4h = get_trend(hma12_4h, ema20_4h)
            trend_d = get_trend(ema20_d, ema50_d)
            trend_w = get_trend(ema20_w, ema50_w)

            # Compter les tendances haussières/baissières pour le score
            score = sum([1 if t == 'Bullish' else -1 if t == 'Bearish' else 0 for t in [trend_1h, trend_4h, trend_d, trend_w]])
            
            results.append({
                'Pair': pair.replace('=X', ''),
                'H1': trend_1h,
                'H4': trend_4h,
                'D': trend_d,
                'W': trend_w,
                'Score': score
            })
            st.info(f"Tendances pour {pair}: H1={trend_1h}, H4={trend_4h}, D={trend_d}, W={trend_w}, Score={score}")
        except Exception as e:
            st.error(f"Erreur générale pour la paire {pair}: {str(e)}")
            # Afficher plus de détails sur l'erreur pour le débogage
            import traceback
            st.error(traceback.format_exc())
            continue # Passer à la paire suivante
    
    if not results:
        st.error("Aucune donnée valide n'a pu être récupérée ou analysée pour les paires Forex. Vérifiez la connexion, les tickers, ou les logs d'erreur ci-dessus.")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    return df

# Interface Streamlit
st.set_page_config(layout="wide") # Utiliser toute la largeur
st.title("Classement des Paires Forex par Tendance MTF")
st.write("Basé sur l'indicateur HMA/EMA pour les timeframes H1, H4, D, et W. Les périodes de données ont été augmentées pour plus de robustesse.")

# Bouton pour actualiser les données
if st.button("Actualiser et Analyser les Paires Forex"):
    with st.spinner("Analyse des données en cours... Merci de patienter."):
        df_results = analyze_forex_pairs() # Renommer pour éviter confusion avec pd
        
        if not df_results.empty:
            st.subheader("Classement des paires Forex")
            
            def style_trends(val):
                color = 'grey' # Neutre par défaut
                if val == 'Bullish':
                    color = 'lightgreen'
                elif val == 'Bearish':
                    color = 'lightcoral'
                return f'background-color: {color}'

            styled_df = df_results.style.applymap(style_trends, subset=['H1', 'H4', 'D', 'W'])
            st.dataframe(styled_df, use_container_width=True)

            st.subheader("Résumé des Indicateurs")
            st.markdown("""
            - **H1 (1 Heure)**: Tendance basée sur HMA(12) vs EMA(20).
            - **H4 (4 Heures)**: Tendance basée sur HMA(12) vs EMA(20).
            - **D (Journalier)**: Tendance basée sur EMA(20) vs EMA(50).
            - **W (Hebdomadaire)**: Tendance basée sur EMA(20) vs EMA(50).
            - **Score**: Somme des tendances (+1 pour Bullish, -1 pour Bearish, 0 pour Neutral) sur les 4 timeframes. Un score plus élevé indique une confluence haussière plus forte.
            """)
        else:
            st.warning("Aucun résultat à afficher. Essayez de réactualiser ou vérifiez les logs d'erreur ci-dessus si des problèmes ont été signalés.")
else:
    st.info("Cliquez sur le bouton ci-dessus pour lancer l'analyse.")
           
