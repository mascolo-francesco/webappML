import joblib
import pandas as pd
import numpy as np

# Carica i modelli
try:
    rf_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    print("✅ Modelli caricati con successo!")
except FileNotFoundError as e:
    print(f"❌ Errore nel caricamento dei modelli: {e}")
    exit(1)

def analyze_model():
    print("\n=== ANALISI DEL MODELLO RANDOM FOREST ===")
    print(f"Numero di alberi: {rf_model.n_estimators}")
    print(f"Profondità massima: {rf_model.max_depth}")
    print(f"Min samples split: {rf_model.min_samples_split}")
    print(f"Min samples leaf: {rf_model.min_samples_leaf}")
    print(f"Max features: {rf_model.max_features}")
    print(f"Bootstrap: {rf_model.bootstrap}")
    
    # Feature importance
    if hasattr(rf_model, 'feature_importances_'):
        FEATURE_COLUMNS = [
            'Age', 'Gender', 'Grade_Level', 'Strength_Score', 'Endurance_Score',
            'Flexibility_Score', 'Speed_Agility_Score', 'BMI', 
            'Health_Fitness_Knowledge_Score', 'Skills_Score', 
            'Class_Participation_Level', 'Attendance_Rate', 'Motivation_Level',
            'Overall_PE_Performance_Score', 'Improvement_Rate', 'Final_Grade',
            'Previous_Semester_PE_Grade', 'Hours_Physical_Activity_Per_Week'
        ]
        
        importances = rf_model.feature_importances_
        feature_importance = list(zip(FEATURE_COLUMNS, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\n=== FEATURE IMPORTANCE (Top 10) ===")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"{i+1}. {feature}: {importance:.4f}")
        
        # Verifica se ci sono feature dominanti (possibile causa di overfitting)
        top_feature_importance = feature_importance[0][1]
        if top_feature_importance > 0.5:
            print(f"\n⚠️  ATTENZIONE: La feature '{feature_importance[0][0]}' ha importanza molto alta ({top_feature_importance:.4f})")
            print("   Questo potrebbe causare overfitting e confidenze al 100%")

def analyze_label_encoders():
    print("\n=== ANALISI LABEL ENCODERS ===")
    for col, encoder in label_encoders.items():
        print(f"{col}: {list(encoder.classes_)}")

def test_predictions_with_variations():
    print("\n=== TEST PREDIZIONI CON VARIAZIONI ===")
    
    FEATURE_COLUMNS = [
        'Age', 'Gender', 'Grade_Level', 'Strength_Score', 'Endurance_Score',
        'Flexibility_Score', 'Speed_Agility_Score', 'BMI', 
        'Health_Fitness_Knowledge_Score', 'Skills_Score', 
        'Class_Participation_Level', 'Attendance_Rate', 'Motivation_Level',
        'Overall_PE_Performance_Score', 'Improvement_Rate', 'Final_Grade',
        'Previous_Semester_PE_Grade', 'Hours_Physical_Activity_Per_Week'
    ]
    
    # Dati base per il test
    base_data = {
        'Age': 16, 
        'Gender': 'Male', 
        'Grade_Level': 10,
        'Strength_Score': 75, 
        'Endurance_Score': 75, 
        'Flexibility_Score': 75,
        'Speed_Agility_Score': 75, 
        'BMI': 22.5, 
        'Health_Fitness_Knowledge_Score': 75,
        'Skills_Score': 75, 
        'Class_Participation_Level': 'High',
        'Attendance_Rate': 90, 
        'Motivation_Level': 'High',
        'Overall_PE_Performance_Score': 75, 
        'Improvement_Rate': 5.0,
        'Final_Grade': 'B', 
        'Previous_Semester_PE_Grade': 'B',
        'Hours_Physical_Activity_Per_Week': 5
    }
    
    print("Testando variazioni per vedere se la confidenza cambia...")
    
    # Test con diversi scenari
    scenarios = [
        ("Studente Eccellente", {**base_data, 'Strength_Score': 95, 'Endurance_Score': 95, 'Overall_PE_Performance_Score': 95, 'Final_Grade': 'A', 'Motivation_Level': 'High'}),
        ("Studente Medio", {**base_data, 'Strength_Score': 70, 'Endurance_Score': 70, 'Overall_PE_Performance_Score': 70, 'Final_Grade': 'C', 'Motivation_Level': 'Medium'}),
        ("Studente Scarso", {**base_data, 'Strength_Score': 40, 'Endurance_Score': 40, 'Overall_PE_Performance_Score': 40, 'Final_Grade': 'D', 'Motivation_Level': 'Low'}),
        ("Caso Borderline 1", {**base_data, 'Strength_Score': 65, 'Endurance_Score': 55, 'Overall_PE_Performance_Score': 60, 'Final_Grade': 'C', 'Motivation_Level': 'Medium'}),
        ("Caso Borderline 2", {**base_data, 'Strength_Score': 85, 'Endurance_Score': 75, 'Overall_PE_Performance_Score': 80, 'Final_Grade': 'B', 'Motivation_Level': 'High'})
    ]
    
    for scenario_name, test_data in scenarios:
        try:
            # Prepara i dati
            student_data = [test_data[col] for col in FEATURE_COLUMNS]
            student_df = pd.DataFrame([student_data], columns=FEATURE_COLUMNS)
            
            # Applica encoding
            for col in label_encoders:
                if col != "Performance" and col in student_df.columns:
                    le = label_encoders[col]
                    student_df[col] = student_df[col].apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    student_df[col] = le.transform(student_df[col])
            
            # Scaling e predizione
            student_scaled = scaler.transform(student_df)
            prediction = rf_model.predict(student_scaled)
            prediction_proba = rf_model.predict_proba(student_scaled)
            
            # Decodifica risultato
            performance_label = label_encoders["Performance"].inverse_transform(prediction)[0]
            max_confidence = max(prediction_proba[0])
            
            print(f"\n{scenario_name}:")
            print(f"  Predizione: {performance_label}")
            print(f"  Confidenza: {max_confidence:.4f} ({max_confidence*100:.2f}%)")
            print(f"  Probabilità per classe: {dict(zip(label_encoders['Performance'].classes_, prediction_proba[0]))}")
            
            if max_confidence >= 0.99:
                print("  ⚠️  CONFIDENZA TROPPO ALTA!")
                
        except Exception as e:
            print(f"  ❌ Errore in {scenario_name}: {e}")

def check_model_trees_consistency():
    print("\n=== ANALISI CONSISTENZA ALBERI ===")
    
    # Test con un dato semplice
    test_data = {
        'Age': 16, 'Gender': 'Male', 'Grade_Level': 10,
        'Strength_Score': 75, 'Endurance_Score': 75, 'Flexibility_Score': 75,
        'Speed_Agility_Score': 75, 'BMI': 22.5, 'Health_Fitness_Knowledge_Score': 75,
        'Skills_Score': 75, 'Class_Participation_Level': 'High',
        'Attendance_Rate': 90, 'Motivation_Level': 'High',
        'Overall_PE_Performance_Score': 75, 'Improvement_Rate': 5.0,
        'Final_Grade': 'B', 'Previous_Semester_PE_Grade': 'B',
        'Hours_Physical_Activity_Per_Week': 5
    }
    
    FEATURE_COLUMNS = list(test_data.keys())
    student_data = [test_data[col] for col in FEATURE_COLUMNS]
    student_df = pd.DataFrame([student_data], columns=FEATURE_COLUMNS)
    
    # Encoding
    for col in label_encoders:
        if col != "Performance" and col in student_df.columns:
            le = label_encoders[col]
            student_df[col] = student_df[col].apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            student_df[col] = le.transform(student_df[col])
    
    student_scaled = scaler.transform(student_df)
    
    # Ottieni predizioni da ogni albero
    tree_predictions = []
    for tree in rf_model.estimators_:
        pred = tree.predict(student_scaled)[0]
        tree_predictions.append(pred)
    
    # Analizza consenso tra alberi
    unique_predictions = np.unique(tree_predictions)
    print(f"Numero di predizioni diverse tra gli alberi: {len(unique_predictions)}")
    
    for pred in unique_predictions:
        count = np.sum(np.array(tree_predictions) == pred)
        percentage = count / len(tree_predictions) * 100
        class_name = label_encoders["Performance"].inverse_transform([pred])[0]
        print(f"  {class_name}: {count}/{len(tree_predictions)} alberi ({percentage:.1f}%)")
    
    if len(unique_predictions) == 1:
        print("⚠️  TUTTI gli alberi concordano sulla stessa predizione!")
        print("   Questo spiega la confidenza al 100%")

if __name__ == "__main__":
    analyze_model()
    analyze_label_encoders()
    test_predictions_with_variations()
    check_model_trees_consistency()