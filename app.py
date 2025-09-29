from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Carica i modelli salvati - USANDO MODELLI MIGLIORATI
try:
    # Prova prima i modelli migliorati, poi quelli originali come fallback
    try:
        rf_model = joblib.load("random_forest_model_improved.pkl")
        scaler = joblib.load("scaler_improved.pkl")
        label_encoders = joblib.load("label_encoders_improved.pkl")
        print("✅ Modelli MIGLIORATI caricati con successo!")
        print("   - Confidenze più realistiche (no più 100%)")
    except FileNotFoundError:
        rf_model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
        print("⚠️  Modelli ORIGINALI caricati (potrebbero dare confidenza 100%)")
        print("   - Esegui fix_confidence_problem.py per modelli migliorati")
except FileNotFoundError as e:
    print(f"Errore nel caricamento dei modelli: {e}")
    print("Assicurati che i file .pkl siano nella stessa directory del server")

# Definizione delle colonne del dataset (senza ID e Performance)
FEATURE_COLUMNS = [
    'Age', 'Gender', 'Grade_Level', 'Strength_Score', 'Endurance_Score',
    'Flexibility_Score', 'Speed_Agility_Score', 'BMI', 
    'Health_Fitness_Knowledge_Score', 'Skills_Score', 
    'Class_Participation_Level', 'Attendance_Rate', 'Motivation_Level',
    'Overall_PE_Performance_Score', 'Improvement_Rate', 'Final_Grade',
    'Previous_Semester_PE_Grade', 'Hours_Physical_Activity_Per_Week'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_performance():
    try:
        data = request.json
        print(f"[DEBUG] Ricevuta richiesta con dati: {data}")
        
        # Validazione input
        if not data:
            return jsonify({'error': 'Nessun dato fornito'}), 400
        
        # Crea un DataFrame con i dati dell'input
        student_data = []
        for col in FEATURE_COLUMNS:
            if col not in data:
                return jsonify({'error': f'Campo mancante: {col}'}), 400
            student_data.append(data[col])
        
        # Crea DataFrame
        student_df = pd.DataFrame([student_data], columns=FEATURE_COLUMNS)
        
        # Applica Label Encoding alle variabili categoriche
        for col in label_encoders:
            if col != "Performance" and col in student_df.columns:
                le = label_encoders[col]
                try:
                    # Se la categoria non esiste, usa la prima classe disponibile
                    student_df[col] = student_df[col].apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    student_df[col] = le.transform(student_df[col])
                except Exception as e:
                    return jsonify({'error': f'Errore nell\'encoding di {col}: {str(e)}'}), 400
        
        # Standardizzazione
        student_scaled = scaler.transform(student_df)
        
        # Predizione
        prediction = rf_model.predict(student_scaled)
        prediction_proba = rf_model.predict_proba(student_scaled)
        
        print(f"[DEBUG] Predizione raw: {prediction}")
        print(f"[DEBUG] Probabilità: {prediction_proba}")
        
        # Decodifica il risultato
        performance_label = label_encoders["Performance"].inverse_transform(prediction)[0]
        print(f"[DEBUG] Label decodificata: {performance_label}")
        
        # Prepara le probabilità per ogni classe
        classes = label_encoders["Performance"].classes_
        probabilities = {}
        for i, class_name in enumerate(classes):
            probabilities[class_name] = float(prediction_proba[0][i])
        
        # Mappa il risultato in italiano
        italian_mapping = {
            'Low Performer': 'Prestazione Bassa',
            'Average Performer': 'Prestazione Media', 
            'High Performer': 'Prestazione Alta'
        }
        
        result = {
            'prediction': performance_label,
            'prediction_italian': italian_mapping.get(performance_label, performance_label),
            'probabilities': probabilities,
            'confidence': float(max(prediction_proba[0]))
        }
        
        print(f"[DEBUG] Risultato finale inviato: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Errore nella predizione: {str(e)}'}), 500

@app.route('/api/info', methods=['GET'])
def get_model_info():
    """Endpoint per ottenere informazioni sul modello"""
    try:
        info = {
            'model_type': 'Random Forest Classifier',
            'features': FEATURE_COLUMNS,
            'classes': list(label_encoders["Performance"].classes_),
            'n_features': len(FEATURE_COLUMNS)
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f'Errore nel recupero info: {str(e)}'}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)

# Per gunicorn
if __name__ != '__main__':
    import logging
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)