# Fonction pour récupérer les données Twelve Data et les formater
def get_twelvedata_data(symbol, interval, output_size):
    try:
        ts = td.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=output_size,
            timezone="Exchange" 
        )
        df = ts.as_pandas()

        if df is None or df.empty:
            st.warning(f"Aucune donnée Twelve Data retournée pour {symbol} ({interval})")
            return pd.DataFrame()

        # Colonnes attendues par le reste du code
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close'
            # 'volume': 'Volume' # On va gérer le volume séparément
        }
        
        # Renommer les colonnes existantes
        df.rename(columns=column_mapping, inplace=True)

        # Gérer la colonne Volume spécifiquement et optionnellement
        final_columns = ['Open', 'High', 'Low', 'Close']
        if 'volume' in df.columns: # Vérifier si la colonne 'volume' (minuscule) existe
            df.rename(columns={'volume': 'Volume'}, inplace=True) # Renommer en 'Volume' (majuscule)
            final_columns.append('Volume')
        elif 'Volume' in df.columns: # Au cas où elle serait déjà en majuscule pour une raison
             final_columns.append('Volume')
        # Si ni 'volume' ni 'Volume' n'existe, on ne l'ajoute pas et on n'aura pas d'erreur

        df.sort_index(ascending=True, inplace=True)

        # S'assurer que toutes les colonnes de base (OHLC) sont présentes après renommage
        required_ohlc = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_ohlc):
            st.error(f"Colonnes OHLC manquantes après traitement pour {symbol} ({interval}). Colonnes disponibles: {df.columns}")
            return pd.DataFrame()

        return df[final_columns] # Sélectionner les colonnes existantes

    except Exception as e:
        st.warning(f"Erreur lors de la récupération des données Twelve Data pour {symbol} ({interval}): {e}")
        # Pour déboguer plus facilement l'erreur exacte de l'API ou du DataFrame:
        import traceback
        st.warning(traceback.format_exc())
        return pd.DataFrame()
       
