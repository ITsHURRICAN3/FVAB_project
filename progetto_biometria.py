import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. CARICAMENTO DEI DATI
df0 = pd.read_csv("Euroleague_2021-2022.csv")
df00 = pd.read_csv("Euroleague_2022-2023.csv")
df1 = pd.read_csv("Euroleague_2023-2024.csv")
df2 = pd.read_csv("Euroleague_2024-2025.csv")
df3 = pd.read_csv("Euroleague_2025-2026.csv")

# Concateniamo i 5 dataset in un unico DataFrame
df = pd.concat([df0, df00, df1, df2, df3], ignore_index=True)

# 2. PREPARAZIONE DELLA STORIA E DEL TARGET
df = df.sort_values(by=['PLAYER','SEASON'])

# Creiamo le statistiche "Per Game" (A Partita)
df['MPG'] = df['MIN'] / df['GP']
df['PPG'] = df['PTS'] / df['GP']

# NOVITÀ: Creiamo le "Lag Features" usando le statistiche "Per Game"
df['PREVIOUS_PPG'] = df.groupby('PLAYER')['PPG'].shift(1)
df['PREVIOUS_MPG'] = df.groupby('PLAYER')['MPG'].shift(1)

# Riempiamo i vuoti per gli esordienti copiando le stats attuali
df['PREVIOUS_PPG'] = df['PREVIOUS_PPG'].fillna(df['PPG'])
df['PREVIOUS_MPG'] = df['PREVIOUS_MPG'].fillna(df['MPG'])

# Creiamo la colonna target "NEXT_PPG" (i Punti a Partita che farà l'anno SUCCESSIVO)
df['NEXT_PPG'] = df.groupby('PLAYER')['PPG'].shift(-1)

# Rimuoviamo le righe dove non conosciamo il target futuro (stagione 25-26)
df_pulito = df.dropna(subset=['NEXT_PPG'])

# Filtro di pulizia (Dataset Cleaning)
# Teniamo solo le stagioni in cui il giocatore ha giocato almeno 120 minuti totali
df_pulito = df_pulito[df_pulito['MIN'] >= 120]

# --- ESPORTAZIONE PER ISPEZIONE ---
# Salviamo il dataset finale e pulito in un nuovo file CSV per permetterti di controllarlo
df_pulito.to_csv("Dataset_Pulito_Pre_Addestramento.csv", index=False)
print("-> File 'Dataset_Pulito_Pre_Addestramento.csv' generato con successo.\n")

# 3. SELEZIONE DELLE FEATURES (Aggiunto AGE)
feature_cols = ["AGE", "MPG", "PPG", "PREVIOUS_PPG", "PREVIOUS_MPG", "TRB", "AST", "BLK", "ST", "TOV", "PF", "FTM", "3PM", "GP", "W"]   

# 4. SUDDIVISIONE IN TRAIN E VALIDATION
df_train = df_pulito[df_pulito['SEASON'].isin(['2021-2022', '2022-2023', '2023-2024'])]
X_train = df_train[feature_cols]
y_train = df_train['NEXT_PPG']

df_val = df_pulito[df_pulito['SEASON'] == '2024-2025']
X_val = df_val[feature_cols]
y_val = df_val['NEXT_PPG']

# 5. ADDESTRAMENTO E CONFRONTO DEI MODELLI
modelli = {
    "Regressione Lineare": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=0, 
        n_estimators=150,     
        learning_rate=0.05,   
        max_depth=4           
    )
}

# Scegliamo un giocatore a caso tra quelli validi per il test 
# (cioè presenti nel set di validazione 24-25 con minuti >= 120)
giocatori_validi = df_val['PLAYER'].unique()
giocatore_scelto = np.random.choice(giocatori_validi)

# Eseguiamo il ciclo per addestrare e valutare entrambi i modelli
for nome_modello, modello in modelli.items():
    # Addestramento
    modello.fit(X_train, y_train)
    
    # 6. VALIDAZIONE E CALCOLO DELLE METRICHE
    predizioni_val = modello.predict(X_val)
    mae = mean_absolute_error(y_val, predizioni_val)
    mse = mean_squared_error(y_val, predizioni_val)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, predizioni_val)
    
    # Calcolo Accuracy in percentuale (basata su WMAPE)
    somma_errori = np.sum(np.abs(y_val - predizioni_val))
    somma_punti_reali = np.sum(y_val)
    accuracy_globale = 100 - ((somma_errori / somma_punti_reali) * 100) if somma_punti_reali > 0 else 0
    
    std_target = np.std(y_val)
    
    print(f"=== PAGELLE GLOBALI: {nome_modello.upper()} ===")
    print(f"Deviazione Std Target:        {std_target:.2f} punti a partita")
    print(f"MAE (Errore Medio):           {mae:.2f} punti a partita")
    print(f"RMSE (Errore Medio Radicale): {rmse:.2f} punti a partita")
    print(f"R-quadro (Precisione R²):     {r2:.3f} ({(r2*100):.1f}%)")
    print(f"Accuracy Globale:             {accuracy_globale:.1f}%\n")
    
    # 7. FOCUS SU UN GIOCATORE
    print(f"--- TEST SU {giocatore_scelto.upper()} ({nome_modello}) ---")
    giocatore_val = df_val[df_val["PLAYER"] == giocatore_scelto]
    
    if not giocatore_val.empty:
        X_giocatore = giocatore_val[feature_cols]
        y_reale = giocatore_val['NEXT_PPG'].values[0]
        predizione = modello.predict(X_giocatore)[0]
        pts_passati = giocatore_val['PREVIOUS_PPG'].values[0]
        pts_correnti = giocatore_val['PPG'].values[0]
        
        print(f"Storico letto: {pts_passati:.1f} PPG (anno prima) -> {pts_correnti:.1f} PPG (quest'anno)")
        print(f"PREVISTI: {predizione:.1f} PPG  |  REALI: {y_reale:.1f} PPG")
        
        errore_singolo = abs(predizione - y_reale)
        acc_singola = max(0, 100 - (errore_singolo / y_reale * 100)) if y_reale > 0 else 0
        print(f"Errore: {errore_singolo:.1f} punti a partita (Accuracy Specifica: {acc_singola:.1f}%)")
    else:
        print(f"Dati non sufficienti per il test.")
    
    print("="*60 + "\n")