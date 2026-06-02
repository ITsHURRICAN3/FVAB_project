# Data-Driven Basketball: Modelli Predittivi per l'Impatto Futuro dei Giocatori
*Presentazione del Progetto di Ricerca*

---

## Slide 1: Titolo e Contesto della Ricerca
### Data-Driven Basketball: Modelli Predittivi per l’Impatto Futuro dei Giocatori
Questo progetto di ricerca, intitolato "Data-Driven Basketball: Modelli Predittivi per l’Impatto Futuro dei Giocatori" e sviluppato per il corso di Fondamenti di Visione Artificiale e Biometria presso l'Università degli Studi di Salerno dai candidati Simone Visconti e Vincenzo Goffredo, si propone di introdurre un framework analitico basato sull'apprendimento supervisionato. L'obiettivo centrale è prevedere in modo quantitativo ed oggettivo la futura produttività offensiva dei giocatori NBA, espressa attraverso i punti medi a partita della stagione successiva, superando le componenti soggettive dello scouting tradizionale.

---

## Slide 2: Introduzione e Motivazioni
### Perché prevedere le performance individuali?
Nel basket professionistico moderno, la sport analytics ha assunto un ruolo strategico fondamentale per la gestione delle franchigie, influenzando la costruzione dei roster, le scelte al Draft e la rinegoziazione dei contratti all'interno del rigido tetto salariale (salary cap). Tra tutte le statistiche individuali, la capacità di realizzare punti (PPG) è la variabile offensiva più determinante sia per il successo della squadra sul campo sia per il valore economico e commerciale dei singoli atleti. Riuscire a proiettare questo valore su scala stagionale consente alle dirigenze di pianificare gli investimenti finanziari a lungo termine basandosi su solide stime quantitative del rendimento futuro.

---

## Slide 3: Definizione del Problema
### Formulazione del task predittivo
Il problema viene formalizzato come un task di regressione supervisionata in cui il modello deve apprendere la relazione che lega il rendimento complessivo di un atleta alla sua produttività nella stagione successiva. La variabile target da stimare è la media punti a partita della stagione successiva ($NEXT\_PPG$). A differenza dei modelli di previsione a breve termine, fortemente influenzati da variabili volatili come i singoli accoppiamenti difensivi o la stanchezza derivante dai viaggi, il nostro modello analizza l'andamento a livello stagionale per catturare la stabilità e la consistenza del rendimento del giocatore.

---

## Slide 4: Le Sfide del Forecasting Sportivo
### Complessità biologiche e rumore nei dati
La stima delle performance future deve fare i conti con complesse dinamiche biologiche e anomalie nei dati storici. In primo luogo, lo sviluppo atletico non segue un andamento lineare, ma è regolato da una curva di invecchiamento (aging curve) il cui picco fisico e prestazionale è stimato attorno ai 27 anni. In secondo luogo, i frequenti scambi di mercato a metà stagione (trades) frammentano le statistiche dei giocatori in righe multiple, rendendo necessarie aggregazioni ponderate sui minuti giocati. Infine, per evitare che infortuni stagionali gravi o la totale assenza di dati storici per i debuttanti (rookies) polarizzino l'addestramento, il framework applica rigorosi filtri di esclusione e algoritmi di imputazione dei dati mancanti.

---

## Slide 5: Pipeline di Integrazione e Preprocessing
### Consolidamento del Dataset e filtraggio degli outlier
Il dataset finale è il risultato dell'integrazione a livello di riga di tre sorgenti CSV stagionali contenenti le statistiche di tiro (scoring), le caratteristiche fisiche (biometrics) e i box-score tradizionali (traditional). L'altezza originariamente espressa in pollici viene convertita in centimetri per standardizzazione e le entry dei giocatori scambiati vengono consolidate calcolando medie ponderate sulle partite giocate. Al fine di eliminare il rumore statistico causato da piccoli campioni di dati o da gravi infortuni, vengono esclusi tutti i giocatori con meno di 120 minuti o 12 partite nella stagione corrente, ed esclusi quelli che hanno disputato meno di 20 partite nella stagione target.

---

## Slide 6: Feature Engineering e Traiettoria Biometrica
### Modellare la maturazione atletica e la continuità
Per consentire al modello di interpretare la fase di carriera in cui si trova l'atleta, è stata introdotta la feature *Peak Age Distance*, che calcola la distanza assoluta dall'età di picco dei 27 anni. Inoltre, vengono introdotte variabili di lag storico relative alle due stagioni precedenti per dare stabilità temporale alle stime. Infine, viene calcolato lo *Scoring Momentum*, rappresentato dalla pendenza lineare dei punti a partita calcolata su un intervallo mobile di tre anni. Questa feature indica se la traiettoria recente del giocatore è in fase di crescita, declino o stabilizzazione, offrendo una preziosa informazione di trend che previene l'effetto di leakage.

***

**[INSERIRE QUI - FIGURA 1: aging_curve.png]**
*Didascalia consigliata: Curva fisiologica dei punti medi a partita (PPG) dei giocatori NBA in funzione dell'età, che mostra il raggiungimento del picco prestazionale a 27 anni.*

***

---

## Slide 7: I Modelli di Machine Learning
### Analisi comparativa degli algoritmi
Il framework implementa e confronta cinque diversi algoritmi di regressione per individuare la struttura predittiva ottimale. La regressione lineare (Linear Regression), applicata su dati standardizzati, costituisce la baseline di riferimento. I modelli basati su ensemble comprendono l'algoritmo Random Forest (che sfrutta il bagging di alberi decisionali) e tre varianti di boosting sequenziale: il Gradient Boosting Regressor tradizionale, l'algoritmo LightGBM (ottimizzato per alberi poco profondi e alta efficienza computazionale) e il regressore CatBoost (strutturato per prevenire l'overfitting su dati tabulari complessi). La sintonizzazione fine degli iperparametri è stata condotta per via sistematica mediante GridSearch.

---

## Slide 8: Risultati Sperimentali
### Valutazione quantitativa sul validation set 2023-24
La valutazione sul validation set della stagione 2023-24 evidenzia che tutti i modelli presentano un'accuratezza eccezionale, riuscendo a stimare la futura media punti dei giocatori entro circa due punti di scarto medio a fronte di una deviazione standard del target pari a 6.66 PPG. Il modello CatBoost ha registrato il miglior fit statistico assoluto con un coefficiente $R^2 = 84.9\%$ e un RMSE minimo di 2.59 PPG. Random Forest ha ottenuto l'errore assoluto più basso (MAE = 2.05 PPG) ed una precisione complessiva dell'81.6\%. Sorprendentemente, la baseline lineare si è rivelata estremamente competitiva ($R^2 = 84.7\%$), confermando che le feature fisiche e storiche ingegnerizzate presentano una forte e coerente linearità con il rendimento futuro.

***

**[INSERIRE QUI - FIGURA 2: model_comparison_metrics.png]**
*Didascalia consigliata: Istogrammi comparativi delle metriche di performance calcolate sui dati di validazione per i 5 modelli.*

***

---

## Slide 9: Analisi delle Predizioni e Regressione verso la Media
### Interpretazione statistica dell'effetto di shrinkage
Analizzando la relazione tra punti reali e punti predetti, emerge chiaramente che per tutti i modelli la pendenza della retta di regressione fit ($m$) è inferiore all'unità (valori compresi tra 0.77 e 0.85). Questo andamento evidenzia il fenomeno matematico dello *shrinkage* e riflette una dinamica sportiva fondamentale: la regressione verso la media. I modelli tendono sistematicamente a sovrastimare i giocatori a basso punteggio e a sottostimare i realizzatori d'élite da oltre 25-30 punti. Questo accade perché le prestazioni eccezionali (sia positive che negative) tendono naturalmente ad attenuarsi nella stagione successiva per effetto del logorio fisico, di accorgimenti difensivi avversari o di normalizzazioni statistiche.

***

**[INSERIRE QUI - FIGURA 3: model_regression_fits.png]**
*Didascalia consigliata: Scatter plot dei valori reali vs predetti per i 5 modelli, che mostrano graficamente l'inclinazione della retta di fit rispetto alla diagonale perfetta y=x.*

***

---

## Slide 10: Importanza delle Feature e Contributo delle Variabili
### Quali fattori influenzano maggiormente le stime?
L'analisi dell'importanza delle variabili rivela che la produttività correntemente registrata ($PPG$) e l'età dell'atleta ($AGE$) rappresentano i fattori trainanti delle predizioni dei modelli. Tuttavia, l'elevato peso assegnato al lag storico dei punti delle stagioni precedenti ($PREV\_PPG$) conferma come lo storico di carriera agisca da stabilizzatore statistico, impedendo al modello di proiettare incrementi o cali irrealistici basati su un singolo anno anomalo. In questo modo, il modello riesce a bilanciare la produttività offensiva attuale con la consistenza storica e l'evoluzione biologica del giocatore.

***

**[INSERIRE QUI - FIGURA 4: feature_importance.png]**
*Didascalia consigliata: Diagramma a barre orizzontali dell'importanza relativa delle feature del modello CatBoost, dominato da PPG correnti e MPG.*

***

---

## Slide 11: Analisi Visiva delle Curve di Regressione
### Modellare la non-linearità del rendimento estremo
Il confronto grafico delle curve di regressione mostra una differenza fondamentale tra i modelli. La regressione lineare ipotizza una crescita costante e illimitata della produttività futura all'aumentare dei punti attuali. Al contrario, le curve stimate da CatBoost, LightGBM e Random Forest mostrano una marcata flessione e curvatura discendente in corrispondenza di valori elevati di PPG corrente, sovrapponendosi in modo preciso alla curva dei dati reali (tratteggiata in nero). Questo dimostra la superiorità strutturale dei modelli basati su alberi decisionali, che riescono a catturare in modo naturale il limite biologico e tattico per cui è statisticamente improbabile che un giocatore mantenga o incrementi medie realizzative estreme anno dopo anno.

***

**[INSERIRE QUI - FIGURA 5: model_regression_curves.png]**
*Didascalia consigliata: Curve di regressione polinomiale di grado 2 dei modelli messe a confronto diretto con la curva di trend dei dati reali.*

***

---

## Slide 12: Limitazioni e Sviluppi Futuri
### I confini del modello e le prospettive di ricerca
Nonostante l'elevato rigore scientifico e l'accuratezza predittiva, il modello proposto risente dell'assenza di dati legati al contesto collettivo della squadra. Variazioni repentine nella guida tecnica, cambiamenti nei sistemi tattici o l'acquisizione di compagni di squadra dominanti (che sottraggono possessi e tiri) rappresentano variabili non catturate dalle statistiche individuali. Per superare questa limitazione, le linee di ricerca future si concentreranno sull'integrazione di metriche di efficienza di squadra e sull'implementazione di reti neurali a grafo (GNN) capaci di mappare e modellare le complesse interazioni e sinergie dinamiche tra i giocatori sul terreno di gioco.

---

## Slide 13: Conclusioni e Contributi del Progetto
### Sintesi dei risultati e utilità pratica
In conclusione, lo studio convalida l'efficacia dell'integrazione di parametri biometrici legati allo sviluppo biologico dell'atleta e di metriche storiche e di momentum per stimare il rendimento offensivo stagionale con scarti estremamente ridotti (~2 PPG). CatBoost e Random Forest si confermano i modelli consigliati, offrendo rispettivamente la massima spiegazione della varianza complessiva ($R^2 = 84.9\%$) e il minor errore assoluto. Questo framework fornisce uno strumento quantitativo, solido e privo di bias soggettivi, a supporto di general manager, scout e analisti sportivi per ottimizzare la pianificazione strategica dei roster e la valutazione dei contratti dei giocatori in NBA.
