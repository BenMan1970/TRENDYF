import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

# Liste des paires Forex à analyser
forex_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X', 'NZDUSD=X']

# Fonction HMA (Hull Moving Average)
def hma(series, length):
    wma1 = series.rolling(window=int(length/2)).mean() * 2
    wma2 = series.rolling(window=length).mean()
    raw_hma = wma1 - wma2
    return raw_hma.rolling(window=int(math.sqrt(length))).mean()

# Fonction pour déterminer la tendance
def get_trend(fast, slow):
    # Vérifier si les séries sont valides et contiennent des données
    if fast.empty or slow.empty:
        return 'Neutral'
    
    # Vérifier si la dernière valeur est NaN
    fast_last = fast.iloc[-1]
    slow_last = slow.iloc[-1]
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
        try:
            # Télécharger les données pour différents timeframes
            data_1h = yf.download(pair, period='1mo', interval='1h', progress=False)
            data_4h = yf.download(pair, period='3mo', interval='4h', progress=False)
            data_d = yf.download(pair, period='1y', interval='1d', progress=False)
            data_w = yf.download(pair, period='5y', interval='1wk', progress=False)

            # Vérifier si les données sont vides
            if data_1h.empty or data_4h.empty or data_d.empty or data_w.empty:
                st.error(f"Données manquantes pour la paire {pair}")
                continue

            # Calculer les moyennes mobiles
            hma12_1h = hma(data_1h['Close'], 12)
            ema20_1h = data_1h['Close'].ewm(span=20, adjust=False).mean()
            hma12_4h = hma(data_4h['Close'], 12)
            ema20_4h = data_4h['Close'].ewm(span=20, adjust=False).mean()
            ema20_d = data_d['Close'].ewm(span=20, adjust=False).mean()
            ema50_d = data_d['Close'].ewm(span=50, adjust=False).mean()
            ema20_w = data_w['Close'].ewm(span=20, adjust=False).mean()
            ema50_w = data_w['Close'].ewm(span=50, adjust=False).mean()

            # Vérifier si les calculs contiennent des NaN
            indicators = [hma12_1h, ema20_1h, hma12_4h, ema20_4h, ema20_d, ema50_d, ema20_w, ema50_w]
            if any(s.isna().iloc[-1] for s in indicators):
                st.error(f"Valeurs manquantes dans les indicateurs pour {pair}")
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
        except Exception as e:
            st.error(f"Erreur pour la paire {pair}: {str(e)}")
            continue
    
    # Créer un DataFrame
    if not results:
        st.error("Aucune donnée valide n'a pu être récupérée pour les paires Forex. Vérifiez la connexion ou les tickers.")
        return pd.DataFrame()  # Retourner un DataFrame vide
    
    df = pd.DataFrame(results)
    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1  # Ajouter le classement
    return df

# Interface Streamlit
st.title("Classement des Paires Forex par Tendance")
st.write("Basé sur l'indicateur MTF (HMA/EMA) pour les timeframes H1, H4, D, et W")

# Bouton pour actualiser les données
if st.button("Actualiser les données"):
    with st.spinner("Analyse des données en cours..."):
        df = analyze_forex_pairs()
        
        # Vérifier si le DataFrame n'est pas vide
        if not df.empty:
            # Afficher le tableau
            st.subheader("Classement des paires Forex")
            styled_df = df.style.apply(
                lambda x: ['background-color: green' if v == 'Bullish' else 'background-color: red' if v == 'Bearish' else 'background-color: blue' for v in x],
                subset=['H1', 'H4', 'D', 'W']
            )
            st.dataframe(styled_df)

            # Résumé
            st.subheader("Résumé")
            st.write("**Paires haussières (score positif)** : Les paires avec un score positif sont plus susceptibles d'être en tendance haussière.")
            st.write("**Paires baissières (score négatif)** : Les paires avec un score négatif sont plus susceptibles d'être en tendance baissière.")
            st.write("**Score** : Calculé en additionnant +1 pour chaque timeframe haussier et -1 pour chaque timeframe baissier.")
        else:
            st.warning("Aucun résultat à afficher. Essayez de réactualiser ou vérifiez les données sources.")
