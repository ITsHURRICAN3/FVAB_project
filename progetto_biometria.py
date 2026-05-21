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
import glob
import os


# unione delle righe in "scoring" e "bio"
scoring_files = sorted(glob.glob("scoring/*.csv"))
dfs = []
for sf in scoring_files:
    bf = sf.replace("scoring\\nba_scoring_", "bio\\nba_bio_").replace("scoring/nba_scoring_", "bio/nba_bio_")
    if os.path.exists(bf):
        s_df = pd.read_csv(sf)
        b_df = pd.read_csv(bf)
        common_cols = list(set(s_df.columns).intersection(b_df.columns))
        merged = pd.merge(s_df, b_df, on=common_cols)
        dfs.append(merged)
    else:
        print(f"Warning: file bio non trovato per {sf}")

df = pd.concat(dfs, ignore_index=True)

# 2. PREPARAZIONE DELLA STORIA E DEL TARGET
# rinomina delle colonne per mantenere coerenza con il dataset vecchio
df['PLAYER'] = df['PLAYER_NAME']
df['SEASON'] = df['_season']
df['NAT'] = df['COUNTRY']
df['HEIGHT'] = df['PLAYER_HEIGHT_INCHES'] * 2.54  # pollici in centimetri
df['MPG'] = df['MIN']  # nei file NBA, MIN rappresenta già i minuti a partita
df['PPG'] = df['PTS']  # e PTS rappresenta già i punti a partita
df['TRB'] = df['REB']  # rimbalzi a partita
df['FG%'] = df['FG_PCT']
df['TS%'] = df['TS_PCT']
df['USG%'] = df['USG_PCT']
df['WIN_RATE'] = df['W_PCT']

# AGGREGAZIONE GIOCATORI SCAMBIATI:
# può accadere che un giocatore venga trasferito in una nuova squadra mid-season, il che comporta la creazione di più righe 
# per lo stesso giocatore nella stessa season. Questa aggregazione serve a prendere i giocatori con più entry per la stessa season 
# e riportare tutti i dati in una sola riga per season tramite somme e media ponderata
agg_weighted = ['MPG', 'PPG', 'TRB', 'AST', 'FG%', 'TS%', 'USG%', 'WIN_RATE',
                'NET_RATING', 'OREB_PCT', 'DREB_PCT', 'AST_PCT', 'AGE', 'HEIGHT']
agg_sum = ['GP', 'W', 'L']
agg_first = ['PLAYER', 'SEASON', 'NAT']

def wavg(col):
    """Restituisce una funzione di media ponderata per GP."""
    return lambda x: np.average(x, weights=df.loc[x.index, 'GP'])

agg_dict = {col: 'first' for col in agg_first}
agg_dict.update({col: wavg(col) for col in agg_weighted})
agg_dict.update({col: 'sum' for col in agg_sum})

df = df.groupby(['PLAYER_ID', 'SEASON'], as_index=False).agg(agg_dict)
df = df.sort_values(by=['PLAYER', 'SEASON']).reset_index(drop=True)

# LAG FEATURES:
# creazione di uno "storico" di varie features per dare al modello una percezione del rendimento passato
df['NEXT_PPG'] = df.groupby('PLAYER')['PPG'].shift(-1)

df['PREV_PPG']   = df.groupby('PLAYER')['PPG'].shift(1) # PPG della stagione passata
df['PREV_PPG_2'] = df.groupby('PLAYER')['PPG'].shift(2) # PPG di due stagioni fa
df['PREV_USG%']  = df.groupby('PLAYER')['USG%'].shift(1) # USG% della stagione passata
df['PREV_GP']    = df.groupby('PLAYER')['GP'].shift(1)   # GP della stagione passata

# fillna per risolvere il problema dei giocatori esordienti: per loro, vengono semplicemente clonati i valori
# della stagione attuale
df['PREV_PPG']   = df['PREV_PPG'].fillna(df['PPG'])
df['PREV_PPG_2'] = df['PREV_PPG_2'].fillna(df['PREV_PPG'])
df['PREV_USG%']  = df['PREV_USG%'].fillna(df['USG%'])
df['PREV_GP']    = df['PREV_GP'].fillna(df['GP'])

# Slope lineare:
# fornisce l'andamento del PPG sulle ultime 3 stagioni. Se è positivo, il giocatore sta migliorando, 
# se è negativo, sta peggiorando.
def compute_slope(series):
    """Calcola la pendenza di una regressione lineare sugli ultimi 3 valori (con shift per evitare leakage)."""
    s = series.shift(1)
    result = pd.Series(index=series.index, dtype=float)
    for i in range(len(s)):
        window = s.iloc[max(0, i-2):i+1].dropna()
        if len(window) >= 2:
            x = np.arange(len(window))
            slope = np.polyfit(x, window.values, 1)[0]
            result.iloc[i] = slope
        else:
            result.iloc[i] = 0.0
    return result

df['PPG_MOMENTUM'] = df.groupby('PLAYER')['PPG'].transform(compute_slope)
df['PPG_MOMENTUM'] = df['PPG_MOMENTUM'].fillna(0.0)

# STAGIONI CONSECUTIVE GIOCATE:
# misura la continuità della carriera di un giocatore. Un giocatore di 40 anni con 30 di esperienza gioca in modo diverso
# da uno di 35 con 5 anni di esperienza
df['CAREER_SEASON_NUM'] = df.groupby('PLAYER').cumcount() + 1

# rimozione delle righe il cui target futuro è sconosciuto (per la predizione)
df_pulito = df.dropna(subset=['NEXT_PPG'])

# DATASET CLEANING:
# dopo aver calcolato i minuti totali, eliminiamo tutti i giocatori che hanno giocato meno di 2h o meno di 12 partite
df_pulito = df_pulito.copy()
df_pulito['TOTAL_MIN'] = df_pulito['MPG'] * df_pulito['GP']
df_pulito = df_pulito[df_pulito['TOTAL_MIN'] >= 120]
df_pulito = df_pulito[df_pulito['GP'] >= 12]

# Peak Age Distance: distanza dall'età di picco atletico (27 anni)
df_pulito['PEAK_AGE_DIST'] = abs(df_pulito['AGE'] - 27)

# 3. SELEZIONE DELLE FEATURES
feature_cols = [
    # Variabili base
    "AGE", "HEIGHT", "MPG", "PPG", "GP", "W",
    # Statistiche di gioco
    "TRB", "AST",
    # Percentuali di tiro
    "FG%", "TS%",
    # Statistiche avanzate
    "USG%", "NET_RATING",
    # Statistiche avanzate di rimbalzo e assist
    "OREB_PCT", "DREB_PCT", "AST_PCT",
    # Features calcolate e contestuali (Lag) 
    "WIN_RATE", "PEAK_AGE_DIST",
    "PREV_PPG", "PREV_PPG_2",
    "PREV_USG%", "PREV_GP",
    # Momentum e continuità di carriera
    "PPG_MOMENTUM", "CAREER_SEASON_NUM"
]

# rimozione di righe con features mancanti/celle vuote
df_pulito = df_pulito.dropna(subset=feature_cols)

# esportazione del dataset come csv per controlli
df_pulito.to_csv("Dataset_Pulito_Pre_Addestramento.csv", index=False)
print("-> File 'Dataset_Pulito_Pre_Addestramento.csv' generato con successo.\n")

# 4. SUDDIVISIONE IN TRAIN E VALIDATION
# tutte le stagioni fino alla 23-24 vengono usate per il traning, mentre 23-24 in coppia con 24-25 viene usata per il test 
df_train = df_pulito[df_pulito['SEASON'] != '2023-24']
X_train = df_train[feature_cols]
y_train = df_train['NEXT_PPG']

df_val = df_pulito[df_pulito['SEASON'] == '2023-24']
X_val = df_val[feature_cols]
y_val = df_val['NEXT_PPG']

# 5. ADDESTRAMENTO E CONFRONTO DEI MODELLI

# HYPERPARAMETER TUNING GridSearchCV
# ricerca degli iperparametri opportuni tramite la libreria sopracitata
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

# scelta di un giocatore a caso per controlli a campione
giocatori_validi = df_val['PLAYER'].unique()
giocatore_scelto = np.random.choice(giocatori_validi)

# ciclo di addestramento dei modelli
for nome_modello, modello in modelli.items():
    # addestramento
    modello.fit(X_train, y_train)
    
    # 6. VALIDAZIONE E CALCOLO DELLE METRICHE
    predizioni_val = modello.predict(X_val)
    mae = mean_absolute_error(y_val, predizioni_val)
    mse = mean_squared_error(y_val, predizioni_val)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, predizioni_val)
    
    # calcolo Accuracy in percentuale (basata su WMAPE)
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