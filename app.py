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

# Fonction pour déterminer la tendance (CORRIGÉE)
def get_trend(fast, slow):
    if fast is None or slow is None or fast.empty or slow.empty:
        return 'Neutral'
    
    fast_dn = fast.dropna()
    slow_dn = slow.dropna()

    fast_last_scalar = np.nan
    if not fast_dn.empty:
        val = fast_dn.iloc[-1]
        if isinstance(val, pd.Series):
            if len(val) == 1: fast_last_scalar = val.item()
            else: return 'Neutral' # Cas anormal
        else:
            fast_last_scalar = val

    slow_last_scalar = np.nan
    if not slow_dn.empty:
        val = slow_dn.iloc[-1]
        if isinstance(val, pd.Series):
            if len(val) == 1: slow_last_scalar = val.item()
            else: return 'Neutral' # Cas anormal
        else:
            slow_last_scalar = val
            
    if pd.isna(fast_last_scalar) or pd.isna(slow_last_scalar):
        return 'Neutral'

    if fast_last_scalar > slow_last_scalar:
        return 'Bullish'
    elif fast_last_scalar < slow_last_scalar:
        return 'Bearish'
    else:
        return 'Neutral'

# Fonction pour récupérer et analyser les données
def analyze_forex_pairs():
    results = []
    
    for pair in forex_pairs:
        st.write(f"Analyse de {pair}...") 
        try:
            data_1h = yf.download(pair, period='10d', interval='1h', progress=False, auto_adjust=False, timeout=10)
            data_4h = yf.download(pair, period='2mo', interval='4h', progress=False, auto_adjust=False, timeout=10)
            data_d = yf.download(pair, period='8mo', interval='1d', progress=False, auto_adjust=False, timeout=10)
            data_w = yf.download(pair, period='3y', interval='1wk', progress=False, auto_adjust=False, timeout=10)

            if data_1h.empty or data_4h.empty or data_d.empty or data_w.empty:
                st.error(f"Données manquantes pour la paire {pair} après téléchargement: H1={len(data_1h)}, H4={len(data_4h)}, D={len(data_d)}, W={len(data_w)}")
                continue

            st.info(f"Données téléchargées pour {pair}: H1={len(data_1h)}, H4={len(data_4h)}, D={len(data_d)}, W={len(data_w)}")

            # S'assurer que la colonne 'Close' existe
            for df_name, df_obj in [("data_1h", data_1h), ("data_4h", data_4h), ("data_d", data_d), ("data_w", data_w)]:
                if 'Close' not in df_obj.columns:
                    st.error(f"La colonne 'Close' est manquante dans {df_name} pour {pair}. Colonnes: {df_obj.columns}")
                    raise KeyError(f"'Close' not found in {df_name}") # Provoque le passage au 'except'

            hma12_1h = hma(data_1h['Close'], 12)
            ema20_1h = data_1h['Close'].ewm(span=20, adjust=False).mean()
            
            hma12_4h = hma(data_4h['Close'], 12)
            ema20_4h = data_4h['Close'].ewm(span=20, adjust=False).mean()
            
            ema20_d = data_d['Close'].ewm(span=20, adjust=False).mean()
            ema50_d = data_d['Close'].ewm(span=50, adjust=False).mean()
            
            ema20_w = data_w['Close'].ewm(span=20, adjust=False).mean()
            ema50_w = data_w['Close'].ewm(span=50, adjust=False).mean()

            indicators = [hma12_1h, ema20_1h, hma12_4h, ema20_4h, ema20_d, ema50_d, ema20_w, ema50_w]
            
            valid_indicators = True
            for ind_series in indicators:
                if ind_series is None or ind_series.dropna().empty:
                    valid_indicators = False
                    st.warning(f"Un indicateur pour {pair} est None ou vide après dropna().")
                    break
            
            if not valid_indicators:
                st.error(f"Valeurs manquantes critiques dans les indicateurs pour {pair}. Un ou plusieurs indicateurs sont entièrement NaN.")
                continue

            trend_1h = get_trend(hma12_1h, ema20_1h)
            trend_4h = get_trend(hma12_4h, ema20_4h)
            trend_d = get_trend(ema20_d, ema50_d)
            trend_w = get_trend(ema20_w, ema50_w)

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
            import traceback
            st.error(traceback.format_exc())
            continue
    
    if not results:
        st.error("Aucune donnée valide n'a pu être récupérée ou analysée pour les paires Forex. Vérifiez la connexion, les tickers, ou les logs d'erreur ci-dessus.")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    return df

# Interface Streamlit
st.set_page_config(layout="wide")
st.title("Classement des Paires Forex par Tendance MTF")
st.write("Basé sur l'indicateur HMA/EMA pour les timeframes H1, H4, D, et W.")

if st.button("Actualiser et Analyser les Paires Forex"):
    with st.spinner("Analyse des données en cours... Merci de patienter."):
        df_results = analyze_forex_pairs()
        
        if not df_results.empty:
            st.subheader("Classement des paires Forex")
            
            def style_trends(val):
                color = 'whitesmoke' 
                if val == 'Bullish': color = 'lightgreen'
                elif val == 'Bearish': color = 'lightcoral'
                elif val == 'Neutral': color = 'lightgoldenrodyellow'
                return f'background-color: {color}'

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
    st.info("Cliquez sur le bouton ci-dessus pour lancer l'analyse.")
           
