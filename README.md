# PE Performance Classifier - Webapp per Classificazione Performance Studenti

Un'applicazione web completa per prevedere le performance degli studenti in educazione fisica utilizzando un modello Random Forest.

## 🚀 Caratteristiche

- **Interface moderna**: Design pulito ispirato a Notion/Vercel
- **Modello ML avanzato**: Random Forest per classificazione delle performance
- **API RESTful**: Server Flask per gestire le predizioni
- **Responsive**: Interfaccia ottimizzata per desktop e mobile
- **Validazione real-time**: Controlli di validazione sui campi input
- **Feedback immediato**: Notifiche e indicatori di stato

## 📋 Struttura del Progetto

```
webapp ML/
├── app.py                           # Server Flask (API backend)
├── templates/
│   └── index.html                   # Template HTML principale
├── static/
│   ├── css/
│   │   └── style.css               # Stili CSS
│   └── js/
│       └── main.js                 # JavaScript frontend
├── es_ML_29_9.ipynb              # Notebook per training del modello
├── student_pe_performance.csv      # Dataset di esempio
├── requirements.txt                # Dipendenze Python
├── random_forest_model.pkl        # Modello Random Forest salvato
├── scaler.pkl                     # StandardScaler salvato
├── label_encoders.pkl             # Label Encoders salvati
└── README.md                      # Questo file
```

## 🛠️ Installazione e Setup

### Prerequisiti

- Python 3.9+
- pip (gestore pacchetti Python)

### Passo 1: Clonare/Scaricare il Progetto

```bash
# Se hai git installato
git clone <repository-url>
cd "webapp ML"

# Oppure scarica e estrai la cartella
```

### Passo 2: Creare un Ambiente Virtuale

```bash
# Crea ambiente virtuale
python3 -m venv .venv

# Attiva l'ambiente virtuale
# Su macOS/Linux:
source .venv/bin/activate

# Su Windows:
# .venv\Scripts\activate
```

### Passo 3: Installare le Dipendenze

```bash
pip install -r requirements.txt
```

### Passo 4: Verificare i File del Modello

Assicurati che questi file siano presenti nella directory principale:
- `random_forest_model.pkl`
- `scaler.pkl`
- `label_encoders.pkl`

Se mancano, esegui il notebook `es_ML_29_9.ipynb` per generarli:

```bash
# Installa jupyter se non presente
pip install jupyter

# Avvia jupyter
jupyter notebook es_ML_29_9.ipynb

# Esegui tutte le celle per generare i file .pkl
```

## 🚀 Avvio dell'Applicazione

### Avvio del Server

```bash
# Assicurati che l'ambiente virtuale sia attivato
source .venv/bin/activate  # su macOS/Linux

# Avvia il server Flask
python app.py
```

Il server sarà disponibile su: **http://127.0.0.1:5001**

### Apertura della Webapp

1. Apri il browser web
2. Vai a: `http://127.0.0.1:5001`
3. Compila il form con i dati dello studente
4. Clicca su "Calcola Previsione"

## 📊 Utilizzo dell'Applicazione

### Campi del Form

L'applicazione richiede le seguenti informazioni sullo studente:

#### Dati Anagrafici
- **Età**: 14-17 anni
- **Genere**: Maschio, Femmina, Altro
- **Anno Scolastico**: Prima, Seconda, Terza, Quarta superiore

#### Valutazioni Fisiche (Punteggi 0-100)
- **Punteggio Forza**
- **Punteggio Resistenza**  
- **Punteggio Flessibilità**
- **Punteggio Velocità/Agilità**
- **BMI**: Indice Massa Corporea (10-50)
- **Punteggio Abilità**
- **Conoscenze Salute e Fitness**

#### Comportamento e Partecipazione
- **Livello Partecipazione**: Basso, Medio, Alto
- **Livello Motivazione**: Basso, Medio, Alto
- **Tasso Presenza**: Percentuale 0-100%
- **Ore Attività Fisica/Settimana**: 0-50 ore

#### Valutazioni Accademiche
- **Punteggio Performance Generale**: 0-100
- **Tasso Miglioramento**: 0-20
- **Voto Finale**: A, B, C
- **Voto Semestre Precedente**: A, B, C

### Risultati

L'applicazione fornisce:
- **Classificazione**: Prestazione Alta, Media o Bassa
- **Livello di Confidenza**: Percentuale di sicurezza della predizione
- **Distribuzione Probabilità**: Probabilità per ogni classe di performance

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
Il modello è stato addestrato su un dataset di studenti con le loro performance in educazione fisica. Le metriche di valutazione sono disponibili nel notebook.

### Preprocessing
- **Encoding Categorico**: Label Encoding per variabili qualitative
- **Standardizzazione**: StandardScaler per variabili numeriche
- **Gestione Missing Values**: Rimozione righe incomplete

## 🚨 Troubleshooting

### Errore: "Modelli non trovati"
Assicurati che i file `.pkl` siano nella directory principale:
```bash
ls -la *.pkl
```
Se mancanti, esegui il notebook per generarli.

### Errore: "Porta 5001 già in uso"
Modifica la porta in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5002)
```

### Errore: "Modulo non trovato"
Verifica che l'ambiente virtuale sia attivato e reinstalla:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Problemi con CSS/JS
Verifica la struttura delle cartelle:
```
static/
├── css/style.css
└── js/main.js
```

## 📝 Sviluppo

### Struttura del Codice

- **Backend**: Flask app in `app.py`
  - Route handling
  - Model loading e prediction
  - API endpoints

- **Frontend**: Vanilla JavaScript in `main.js`
  - Form validation
  - API calls
  - UI interactions

- **Styling**: CSS moderno con variabili
  - Design system coerente
  - Responsive design
  - Accessibilità

### Aggiungere Nuove Features

1. **Nuovi campi input**: Modifica `index.html` e `main.js`
2. **Nuove validazioni**: Aggiorna `validationRules` in `main.js`
3. **Nuovi endpoint**: Aggiungi route in `app.py`

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT. Consulta il file LICENSE per dettagli.

## 🤝 Contributi

I contributi sono benvenuti! Per contribuire:

1. Fork del progetto
2. Crea un branch per la tua feature
3. Commit delle modifiche
4. Push del branch
5. Apri una Pull Request

## 📞 Supporto

Per supporto o domande:
- Apri una issue su GitHub
- Controlla la sezione Troubleshooting
- Verifica la documentazione API

---

**Creato con ❤️ per l'analisi delle performance studentesche in educazione fisica**