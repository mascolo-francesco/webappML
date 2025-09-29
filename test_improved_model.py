import requests
import json

# Test del modello migliorato con diverse tipologie di studenti
test_students = [
    {
        "name": "Studente Eccellente",
        "data": {
            'Age': 16, 'Gender': 'Male', 'Grade_Level': '10th',
            'Strength_Score': 95, 'Endurance_Score': 95, 'Flexibility_Score': 90,
            'Speed_Agility_Score': 90, 'BMI': 22.0, 'Health_Fitness_Knowledge_Score': 95,
            'Skills_Score': 95, 'Class_Participation_Level': 'High',
            'Attendance_Rate': 98, 'Motivation_Level': 'High',
            'Overall_PE_Performance_Score': 95, 'Improvement_Rate': 8.0,
            'Final_Grade': 'A', 'Previous_Semester_PE_Grade': 'A',
            'Hours_Physical_Activity_Per_Week': 8
        }
    },
    {
        "name": "Studente Medio",
        "data": {
            'Age': 15, 'Gender': 'Female', 'Grade_Level': '9th',
            'Strength_Score': 70, 'Endurance_Score': 70, 'Flexibility_Score': 75,
            'Speed_Agility_Score': 70, 'BMI': 21.0, 'Health_Fitness_Knowledge_Score': 70,
            'Skills_Score': 70, 'Class_Participation_Level': 'Medium',
            'Attendance_Rate': 85, 'Motivation_Level': 'Medium',
            'Overall_PE_Performance_Score': 70, 'Improvement_Rate': 3.0,
            'Final_Grade': 'C', 'Previous_Semester_PE_Grade': 'C',
            'Hours_Physical_Activity_Per_Week': 3
        }
    },
    {
        "name": "Studente Scarso",
        "data": {
            'Age': 17, 'Gender': 'Male', 'Grade_Level': '11th',
            'Strength_Score': 45, 'Endurance_Score': 40, 'Flexibility_Score': 50,
            'Speed_Agility_Score': 45, 'BMI': 25.0, 'Health_Fitness_Knowledge_Score': 45,
            'Skills_Score': 45, 'Class_Participation_Level': 'Low',
            'Attendance_Rate': 70, 'Motivation_Level': 'Low',
            'Overall_PE_Performance_Score': 45, 'Improvement_Rate': 1.0,
            'Final_Grade': 'D', 'Previous_Semester_PE_Grade': 'D',
            'Hours_Physical_Activity_Per_Week': 1
        }
    },
    {
        "name": "Caso Borderline",
        "data": {
            'Age': 16, 'Gender': 'Female', 'Grade_Level': '10th',
            'Strength_Score': 65, 'Endurance_Score': 60, 'Flexibility_Score': 70,
            'Speed_Agility_Score': 65, 'BMI': 23.0, 'Health_Fitness_Knowledge_Score': 65,
            'Skills_Score': 65, 'Class_Participation_Level': 'Medium',
            'Attendance_Rate': 80, 'Motivation_Level': 'Medium',
            'Overall_PE_Performance_Score': 65, 'Improvement_Rate': 2.5,
            'Final_Grade': 'C', 'Previous_Semester_PE_Grade': 'B',
            'Hours_Physical_Activity_Per_Week': 4
        }
    }
]

def test_local_model():
    """Testa il modello direttamente (senza server web)"""
    print("=== TEST DIRETTO DEL MODELLO MIGLIORATO ===\n")
    
    import joblib
    import pandas as pd
    
    # Carica i modelli migliorati
    try:
        rf_model = joblib.load("random_forest_model_improved.pkl")
        scaler = joblib.load("scaler_improved.pkl")
        label_encoders = joblib.load("label_encoders_improved.pkl")
        print("✅ Modelli migliorati caricati!\n")
    except FileNotFoundError:
        print("❌ Modelli migliorati non trovati!")
        return
    
    FEATURE_COLUMNS = [
        'Age', 'Gender', 'Grade_Level', 'Strength_Score', 'Endurance_Score',
        'Flexibility_Score', 'Speed_Agility_Score', 'BMI', 
        'Health_Fitness_Knowledge_Score', 'Skills_Score', 
        'Class_Participation_Level', 'Attendance_Rate', 'Motivation_Level',
        'Overall_PE_Performance_Score', 'Improvement_Rate', 'Final_Grade',
        'Previous_Semester_PE_Grade', 'Hours_Physical_Activity_Per_Week'
    ]
    
    for student in test_students:
        print(f"🎓 {student['name']}:")
        
        # Prepara i dati
        student_data = [student['data'][col] for col in FEATURE_COLUMNS]
        student_df = pd.DataFrame([student_data], columns=FEATURE_COLUMNS)
        
        # Encoding
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
        
        # Risultati
        performance_label = label_encoders["Performance"].inverse_transform(prediction)[0]
        confidence = max(prediction_proba[0])
        
        # Mappa italiana
        italian_mapping = {
            'Low Performer': 'Prestazione Bassa',
            'Average Performer': 'Prestazione Media', 
            'High Performer': 'Prestazione Alta'
        }
        
        print(f"   Predizione: {italian_mapping.get(performance_label, performance_label)}")
        print(f"   Confidenza: {confidence:.2f} ({confidence*100:.1f}%)")
        
        # Mostra tutte le probabilità
        classes = label_encoders["Performance"].classes_
        print("   Probabilità dettagliate:")
        for i, class_name in enumerate(classes):
            italian_class = italian_mapping.get(class_name, class_name)
            prob = prediction_proba[0][i]
            print(f"     - {italian_class}: {prob:.3f} ({prob*100:.1f}%)")
        
        # Indicatore di qualità della confidenza
        if confidence >= 0.95:
            print("   ⚠️  Confidenza molto alta")
        elif confidence >= 0.80:
            print("   ✅ Confidenza buona")
        else:
            print("   💡 Confidenza moderata (incertezza)")
        
        print()

if __name__ == "__main__":
    test_local_model()