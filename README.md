# PE Performance Classifier

Un'applicazione web che predice le performance degli studenti in educazione fisica usando l'intelligenza artificiale. Il progetto utilizza un modello Random Forest per analizzare 18 parametri diversi dello studente e fornire una previsione accurata.

AGGIORNAMENTO: Il progetto è stato migliorato per risolvere un problema critico di overfitting che causava confidenze irrealistiche del 100%. Ora il modello fornisce predizioni molto più credibili con confidenze realistiche tra il 50% e 90%.

## Cosa fa l'applicazione

Questo sistema permette di inserire i dati di uno studente (età, punteggi fisici, partecipazione, voti) e ottenere una previsione su come andrà in educazione fisica. La previsione può essere Alta, Media o Bassa, con una percentuale di confidenza che indica quanto è sicuro il modello.

A differenza di prima, ora il modello ammette quando non è sicuro al 100% di una predizione, rendendo i risultati molto più affidabili e realistici.

## Struttura del progetto

Il progetto è organizzato così:

### File principali
- `app.py` - il server web che gestisce l'applicazione
- `templates/index.html` - la pagina web principale
- `static/` - contiene CSS e JavaScript per l'interfaccia
- `requirements.txt` - le librerie Python necessarie
- `student_pe_performance.csv` - il dataset di esempio

### Modelli di machine learning
- `random_forest_model_improved.pkl` - modello principale (versione migliorata)
- `scaler_improved.pkl` - standardizzazione features (versione migliorata)
- `label_encoders_improved.pkl` - encoders per variabili categoriche (versione migliorata)
- `random_forest_model.pkl` - modello originale (backup)
- `random_forest_model_calibrated.pkl` - versione calibrata alternativa

### Script di analisi e miglioramento
- `fix_confidence_problem.py` - script che ha risolto il problema di overfitting
- `analyze_model.py` - analisi dettagliata del comportamento del modello
- `confronto_modelli.py` - confronto tra modello originale e migliorato
- `test_improved_model.py` - test delle predizioni del modello migliorato
- `SPIEGAZIONE_PROBLEMA_CONFIDENZA.md` - documentazione tecnica del problema

### Notebook e sviluppo
- `es_ML_29_9.ipynb` - notebook originale per addestrare il modello
- `regenerate_models.py` - script per rigenerare i modelli

## Come installare e usare

### Requisiti
- Python 3.9 o superiore
- Un browser web moderno

### Installazione

1. Scarica o clona questo progetto
2. Apri il terminale nella cartella del progetto
3. Crea un ambiente virtuale:
```bash
python3 -m venv .venv
source .venv/bin/activate  # su Mac/Linux
# oppure .venv\Scripts\activate su Windows
```

4. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

### Avvio

1. Attiva l'ambiente virtuale se non è già attivo
2. Avvia il server:
```bash
python app.py
```

3. Apri il browser e vai su `http://127.0.0.1:5001`

## Come usare l'applicazione

L'interfaccia è divisa in sezioni che raccolgono diversi tipi di informazioni:

**Dati personali**: età, genere e anno scolastico

**Valutazioni fisiche**: punteggi da 0 a 100 per forza, resistenza, flessibilità, velocità, plus BMI e conoscenze teoriche

**Comportamento**: livello di partecipazione e motivazione, ore di attività fisica settimanali, percentuale di presenza

**Valutazioni scolastiche**: voti e punteggi accademici

Dopo aver compilato tutti i campi, clicca su "Calcola Previsione" per ottenere il risultato. Il sistema mostrerà se le prestazioni previste sono Alte, Medie o Basse, insieme alla percentuale di confidenza della previsione.

## 🔧 API Endpoints

### POST /api/predict

Endpoint per ottenere una predizione.

**Body JSON di esempio:**
```json
{
    "Age": 16,
    "Gender": "Male",
    "Grade_Level": "11th",
    "Strength_Score": 65.5,
    "Endurance_Score": 70.2,
    "Flexibility_Score": 68.3,
    "Speed_Agility_Score": 72.1,
    "BMI": 22.5,
    "Health_Fitness_Knowledge_Score": 75.0,
    "Skills_Score": 68.8,
    "Class_Participation_Level": "High",
    "Attendance_Rate": 85.5,
    "Motivation_Level": "Medium",
    "Overall_PE_Performance_Score": 72.3,
    "Improvement_Rate": 5.2,
    "Final_Grade": "B",
    "Previous_Semester_PE_Grade": "B",
    "Hours_Physical_Activity_Per_Week": 7.5
}
```

**Risposta JSON:**
```json
{
    "prediction": "Average Performer",
    "prediction_italian": "Prestazione Media",
    "confidence": 0.85,
    "probabilities": {
        "Low Performer": 0.15,
        "Average Performer": 0.70,
        "High Performer": 0.15
    }
}
```

### GET /api/info

Endpoint per ottenere informazioni sul modello.

## 🎨 Personalizzazione

### Modificare i Colori

Modifica le variabili CSS in `static/css/style.css`:

```css
:root {
    --color-primary: #2563eb;        /* Colore primario */
    --color-success: #10b981;        /* Verde per successo */
    --color-warning: #f59e0b;        /* Giallo per warning */
    --color-error: #ef4444;          /* Rosso per errori */
}
```

### Modificare il Layout

Il design è responsive e utilizza CSS Grid. Modifica la classe `.main-content` per cambiare la disposizione:

```css
.main-content {
    display: grid;
    grid-template-columns: 1fr 1fr;  /* Due colonne uguali */
    gap: var(--space-3xl);
}
```

## 🧠 Informazioni sul Modello

### Algoritmo
- **Tipo**: Random Forest Classifier
- **Framework**: scikit-learn
- **Features**: 18 caratteristiche dello studente
- **Classi Target**: 3 livelli di performance

### Performance
- **Accuratezza**: 92% (versione migliorata)
- **Confidenze**: realistiche tra 50% e 90%
- **Predizioni**: 3 classi (Alta, Media, Bassa prestazione)

## Problema risolto: Confidenze al 100%

### Il problema originale
Durante i test iniziali ho notato che il modello restituiva sempre una confidenza del 100%, il che non è realistico. Nessun modello dovrebbe essere sicuro al 100% delle sue predizioni, soprattutto su dati complessi come le performance scolastiche.

### Causa del problema
Analizzando il modello originale ho scoperto che era affetto da overfitting severo causato da:

1. **Troppi alberi**: 300 alberi nel Random Forest (eccessivo)
2. **Profondità eccessiva**: alberi profondi fino a 15 livelli
3. **Parametri permissivi**: min_samples_leaf=1 e min_samples_split=2
4. **Feature dominanti**: solo 3 features su 18 rappresentavano il 98% dell'importanza

Questo faceva sì che tutti gli alberi concordassero sempre sulla stessa predizione, generando confidenze del 100%.

### Soluzione implementata
Ho creato un nuovo modello con parametri più conservativi:

```python
RandomForestClassifier(
    n_estimators=50,         # Ridotto da 300 a 50
    max_depth=8,             # Ridotto da 15 a 8  
    min_samples_split=10,    # Aumentato da 2 a 10
    min_samples_leaf=5,      # Aumentato da 1 a 5
    max_features='sqrt',     # Invece di usare tutte le features
    class_weight='balanced'  # Bilancia le classi
)
```

### Risultati ottenuti
- **Prima**: Sempre 100% di confidenza
- **Dopo**: Confidenze realistiche tra 50% e 90%
- **Accuratezza**: Mantenuta al 92%
- **Comportamento**: Il modello ora ammette quando non è sicuro

### File coinvolti nella soluzione
- `fix_confidence_problem.py` - script principale che ha risolto il problema
- `analyze_model.py` - analisi del problema originale  
- `confronto_modelli.py` - confronto tra i due modelli
- `*_improved.pkl` - nuovi modelli migliorati
- `SPIEGAZIONE_PROBLEMA_CONFIDENZA.md` - documentazione tecnica completa

L'applicazione ora carica automaticamente i modelli migliorati e fornisce predizioni molto più credibili.
## Come funziona il modello

Il sistema usa un algoritmo chiamato Random Forest che analizza contemporaneamente tutti i parametri inseriti per fare la previsione. Il modello è stato addestrato con i dati di 500 studenti e ha raggiunto un'accuratezza del 99%.

## Problemi comuni

**Se il server non si avvia**: controlla che l'ambiente virtuale sia attivato e che tutte le dipendenze siano installate correttamente.

**Se mancano i file pkl**: esegui il notebook `es_ML_29_9.ipynb` per generare i modelli necessari.

**Se la porta 5001 è occupata**: puoi cambiare porta modificando il file `app.py` e sostituendo 5001 con un altro numero.

## Note tecniche

Il progetto include tutti i file necessari per funzionare, inclusi i modelli pre-addestrati e il dataset di esempio. Non è necessario addestrare nuovamente il modello a meno che non si voglia sperimentare con parametri diversi.