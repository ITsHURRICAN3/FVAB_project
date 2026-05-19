import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from catboost import CatBoostRegressor

# 1. CARICAMENTO DEI DATI
df0 = pd.read_csv("Euroleague_2021-2022.csv")
df00 = pd.read_csv("Euroleague_2022-2023.csv")
df1 = pd.read_csv("Euroleague_2023-2024.csv")
df2 = pd.read_csv("Euroleague_2024-2025.csv")
df3 = pd.read_csv("Euroleague_2025-2026.csv")

# Concatenare i 5 dataset in un unico DataFrame
df = pd.concat([df0, df00, df1, df2, df3], ignore_index=True)

# 2. PREPARAZIONE DELLA STORIA E DEL TARGET
df = df.sort_values(by=['PLAYER','SEASON'])

# Creazione delle statistiche per game
df['MPG'] = df['MIN'] / df['GP']
df['PPG'] = df['PTS'] / df['GP']

# Creazione della colonna target "NEXT_PPG" (i Punti a Partita che farà l'anno successivo)
df['NEXT_PPG'] = df.groupby('PLAYER')['PPG'].shift(-1)

# Rimozione delle righe il cui target futuro è sconosciuto (stagione 25-26)
df_pulito = df.dropna(subset=['NEXT_PPG'])

# Filtro di pulizia (Dataset Cleaning)
# Teniamo solo i giocatori con almeno 120 minuti totali E almeno 10 partite giocate
# (elimina le stagioni troppo corte che potrebbero essere dovute a infortuni o panchinari estremi)
df_pulito = df_pulito[df_pulito['MIN'] >= 120]
df_pulito = df_pulito[df_pulito['GP'] >= 10]

# --- FEATURE ENGINEERING: Nuove variabili calcolate ---
# Win Rate: normalizza le vittorie rispetto alle partite giocate
df_pulito = df_pulito.copy()
df_pulito['WIN_RATE'] = df_pulito['W'] / df_pulito['GP']

# Peak Age Distance: distanza dall'età di picco atletico (27 anni)
# Un giocatore di 24 anni tende a migliorare, uno di 33 tende a calare
df_pulito['PEAK_AGE_DIST'] = abs(df_pulito['AGE'] - 27)

# Esportazione del dataset per controllo
df_pulito.to_csv("Dataset_Pulito_Pre_Addestramento.csv", index=False)
print("-> File 'Dataset_Pulito_Pre_Addestramento.csv' generato con successo.\n")

# 3. SELEZIONE DELLE FEATURES
feature_cols = [
    # Variabili base
    "AGE", "HEIGHT", "MPG", "PPG", "GP", "W",
    # Statistiche di gioco
    "ORB", "DRB", "TRB", "AST", "BLK", "ST", "TOV", "PF", "FTM", "3PM", "FGM", "FGA",
    # Percentuali di tiro
    "FG%", "3P%", "FT%", "2P%", "eFG%", "TS%",
    # Statistiche avanzate
    "USG%", "EFF", "PER", "+/-", "POSS",
    # Rating di squadra e individuali
    "TM OFF RTG (ON)", "TM DEF RTG (ON)", "TM NET RTG (ON)",
    "IND OFF RTG", "IND DEF RTG", "IND NET RTG",
    # Win Shares e impatto
    "OWS", "DWS", "WS", "BPM", "OBPM", "DBPM", "VORP",
    # Features calcolate
    "WIN_RATE", "PEAK_AGE_DIST"
]

# 4. SUDDIVISIONE IN TRAIN E VALIDATION
df_train = df_pulito[df_pulito['SEASON'].isin(['2021-2022', '2022-2023', '2023-2024'])]
X_train = df_train[feature_cols]
y_train = df_train['NEXT_PPG']

df_val = df_pulito[df_pulito['SEASON'] == '2024-2025']
X_val = df_val[feature_cols]
y_val = df_val['NEXT_PPG']

# 5. ADDESTRAMENTO E CONFRONTO DEI MODELLI

# --- HYPERPARAMETER TUNING con GridSearchCV ---
# Cerchiamo automaticamente i migliori parametri per i modelli più potenti
print("Ricerca degli iperparametri ottimali in corso...")

# Tuning Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}
gs_gb = GridSearchCV(GradientBoostingRegressor(random_state=0), param_grid_gb, cv=3, scoring='r2', n_jobs=-1)
gs_gb.fit(X_train, y_train)
print(f"  -> Gradient Boosting best params: {gs_gb.best_params_} (R² cv: {gs_gb.best_score_:.3f})")

# Tuning LightGBM
param_grid_lgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'num_leaves': [20, 31, 50]
}
gs_lgb = GridSearchCV(lgb.LGBMRegressor(random_state=0, verbose=-1), param_grid_lgb, cv=3, scoring='r2', n_jobs=-1)
gs_lgb.fit(X_train, y_train)
print(f"  -> LightGBM best params: {gs_lgb.best_params_} (R² cv: {gs_lgb.best_score_:.3f})")
print()

modelli = {
    "Regressione Lineare": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting (Tuned)": gs_gb.best_estimator_,
    "LightGBM (Tuned)": gs_lgb.best_estimator_,
    "CatBoost": CatBoostRegressor(random_state=0, verbose=0)
}

# Scelta di un giocatore a caso per il test
giocatori_validi = df_val['PLAYER'].unique()
giocatore_scelto = np.random.choice(giocatori_validi)

# Ciclo per addestrare e valutare i modelli
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
    
    # 7. FOCUS SU UN GIOCATORE CASUALE
    giocatore_val = df_val[df_val["PLAYER"] == giocatore_scelto]
    if not giocatore_val.empty:
        X_giocatore = giocatore_val[feature_cols]
        y_reale = giocatore_val['NEXT_PPG'].values[0]
        predizione = modello.predict(X_giocatore)[0]
        errore_singolo = abs(predizione - y_reale)
        acc_singola = max(0, 100 - (errore_singolo / y_reale * 100)) if y_reale > 0 else 0
        print(f"--- TEST SU {giocatore_scelto.upper()} ---")
        print(f"Previsto: {predizione:.1f} PPG | Reale: {y_reale:.1f} PPG")
        print(f"Errore: {errore_singolo:.1f} punti a partita (Accuracy Specifica: {acc_singola:.1f}%)")

    # 8. ESPORTAZIONE PREDIZIONI IN CSV
    df_risultati = df_val[['PLAYER', 'SEASON']].copy()
    for col in feature_cols:
        df_risultati[col] = X_val[col]
    df_risultati['REALE_NEXT_PPG'] = y_val
    df_risultati['PREDIZIONE'] = np.round(predizioni_val, 1)
    
    nome_file = f"Predizioni_{nome_modello.replace(' ', '_')}.csv"
    df_risultati.to_csv(nome_file, index=False)
    
    print(f"-> Predizioni salvate in '{nome_file}'")
    print("================================================================\n")