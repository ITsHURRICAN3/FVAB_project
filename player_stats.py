import pandas as pd
import argparse

def get_player_stats(player_name, features):
    file_path = "Dataset_Pulito_Pre_Addestramento.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Errore: Il file {file_path} non è stato trovato.")
        return

    # Controlla se il giocatore esiste
    if player_name not in df['PLAYER'].values:
        print(f"Giocatore '{player_name}' non trovato nel dataset.")
        return

    # Filtra per giocatore
    player_df = df[df['PLAYER'] == player_name].copy()

    # Ordina per stagione
    if 'SEASON' in player_df.columns:
        player_df = player_df.sort_values(by='SEASON')

    # Prepara la lista delle features da mostrare
    # Rimuove gli spazi dalle features in input
    feature_list = [f.strip() for f in features.split(',')]
    
    # Aggiungi sempre 'SEASON' alle features se non è presente per chiarezza di output
    display_features = []
    if 'SEASON' not in feature_list:
        display_features.append('SEASON')
    display_features.extend(feature_list)

    # Controlla se le feature richieste esistono
    missing_features = [f for f in display_features if f not in player_df.columns]
    if missing_features:
        print(f"Errore: Le seguenti features non sono presenti nel dataset: {', '.join(missing_features)}")
        return

    # Stampa i risultati
    result_df = player_df[display_features]
    print(f"\nStatistiche per {player_name} ordinate per stagione:\n")
    print(result_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recupera le statistiche di un giocatore ordinate per stagione.")
    parser.add_argument("-p", "--player", type=str, help="Nome del giocatore (es. 'Tonut S.')")
    parser.add_argument("-f", "--features", type=str, help="Features separate da virgola (es. 'NAT, HEIGHT, AGE')")

    args = parser.parse_args()

    if args.player and args.features:
        get_player_stats(args.player, args.features)
    else:
        player_name = input("Inserisci il nome del giocatore (es. 'Tonut S.'): ")
        features = input("Inserisci le features desiderate separate da virgola (es. 'NAT, HEIGHT, AGE'): ")
        if player_name and features:
            get_player_stats(player_name, features)
        else:
            print("Errore: Devi inserire sia il nome del giocatore che le features.")
