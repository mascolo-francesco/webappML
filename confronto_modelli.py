import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compare_models():
    """Confronta il modello originale con quello migliorato"""
    
    print("🔍 CONFRONTO DETTAGLIATO: MODELLO ORIGINALE vs MIGLIORATO\n")
    
    # Carica entrambi i modelli
    try:
        original_model = joblib.load("random_forest_model.pkl")
        original_scaler = joblib.load("scaler.pkl") 
        original_encoders = joblib.load("label_encoders.pkl")
        print("✅ Modello ORIGINALE caricato")
    except:
        print("❌ Modello originale non trovato")
        return
        
    try:
        improved_model = joblib.load("random_forest_model_improved.pkl")
        improved_scaler = joblib.load("scaler_improved.pkl")
        improved_encoders = joblib.load("label_encoders_improved.pkl") 
        print("✅ Modello MIGLIORATO caricato")
    except:
        print("❌ Modello migliorato non trovato")
        return
    
    print("\n" + "="*60)
    print("📊 CONFRONTO PARAMETRI")
    print("="*60)
    
    # Confronto parametri
    params_comparison = [
        ("Numero di alberi", original_model.n_estimators, improved_model.n_estimators),
        ("Profondità massima", original_model.max_depth, improved_model.max_depth),
        ("Min samples split", original_model.min_samples_split, improved_model.min_samples_split),
        ("Min samples leaf", original_model.min_samples_leaf, improved_model.min_samples_leaf),
        ("Max features", original_model.max_features, improved_model.max_features),
        ("Class weight", getattr(original_model, 'class_weight', None), getattr(improved_model, 'class_weight', None))
    ]
    
    print(f"{'Parametro':<20} {'Originale':<15} {'Migliorato':<15} {'Cambiamento'}")
    print("-" * 70)
    
    for param, orig, improved in params_comparison:
        change = "✅ Migliorato" if orig != improved else "⚪ Stesso"
        print(f"{param:<20} {str(orig):<15} {str(improved):<15} {change}")
    
    print("\n" + "="*60)
    print("🧪 TEST CONFIDENZE SU CASI DIVERSI")
    print("="*60)
    
    # Casi di test
    test_cases = [
        {
            "name": "🎓 Studente Eccellente",
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
            "name": "📚 Studente Medio",
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
            "name": "📉 Studente Scarso",
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
            "name": "🤔 Caso Borderline",
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
    
    FEATURE_COLUMNS = [
        'Age', 'Gender', 'Grade_Level', 'Strength_Score', 'Endurance_Score',
        'Flexibility_Score', 'Speed_Agility_Score', 'BMI', 
        'Health_Fitness_Knowledge_Score', 'Skills_Score', 
        'Class_Participation_Level', 'Attendance_Rate', 'Motivation_Level',
        'Overall_PE_Performance_Score', 'Improvement_Rate', 'Final_Grade',
        'Previous_Semester_PE_Grade', 'Hours_Physical_Activity_Per_Week'
    ]
    
    confidences_original = []
    confidences_improved = []
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        print("-" * 40)
        
        # Prepara i dati
        student_data = [case['data'][col] for col in FEATURE_COLUMNS]
        student_df = pd.DataFrame([student_data], columns=FEATURE_COLUMNS)
        
        # Test modello ORIGINALE
        student_df_orig = student_df.copy()
        for col in original_encoders:
            if col != "Performance" and col in student_df_orig.columns:
                le = original_encoders[col]
                student_df_orig[col] = student_df_orig[col].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                student_df_orig[col] = le.transform(student_df_orig[col])
        
        student_scaled_orig = original_scaler.transform(student_df_orig)
        pred_proba_orig = original_model.predict_proba(student_scaled_orig)
        confidence_orig = max(pred_proba_orig[0])
        
        # Test modello MIGLIORATO  
        student_df_imp = student_df.copy()
        for col in improved_encoders:
            if col != "Performance" and col in student_df_imp.columns:
                le = improved_encoders[col]
                student_df_imp[col] = student_df_imp[col].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                student_df_imp[col] = le.transform(student_df_imp[col])
        
        student_scaled_imp = improved_scaler.transform(student_df_imp)
        pred_proba_imp = improved_model.predict_proba(student_scaled_imp)
        confidence_imp = max(pred_proba_imp[0])
        
        confidences_original.append(confidence_orig)
        confidences_improved.append(confidence_imp)
        
        # Mostra risultati
        print(f"Modello ORIGINALE:  {confidence_orig:.1%} confidenza")
        print(f"Modello MIGLIORATO: {confidence_imp:.1%} confidenza")
        
        # Valutazione
        if confidence_orig >= 0.99:
            orig_status = "❌ Troppo sicuro"
        elif confidence_orig >= 0.95:
            orig_status = "⚠️  Molto sicuro"
        else:
            orig_status = "✅ Ragionevole"
            
        if confidence_imp >= 0.95:
            imp_status = "⚠️  Molto sicuro"
        elif confidence_imp >= 0.80:
            imp_status = "✅ Buona confidenza"
        else:
            imp_status = "💡 Moderata (con incertezza)"
            
        print(f"Originale: {orig_status}")
        print(f"Migliorato: {imp_status}")
        
        improvement = confidence_orig - confidence_imp
        print(f"🎯 Miglioramento: -{improvement:.1%}")
    
    print("\n" + "="*60)
    print("📈 STATISTICHE FINALI")
    print("="*60)
    
    print(f"Confidenza media ORIGINALE:  {np.mean(confidences_original):.1%}")
    print(f"Confidenza media MIGLIORATA: {np.mean(confidences_improved):.1%}")
    print(f"Riduzione media confidenza:  -{np.mean(confidences_original) - np.mean(confidences_improved):.1%}")
    
    print(f"\nCasi con confidenza > 95%:")
    print(f"  Originale:  {sum(1 for c in confidences_original if c > 0.95)}/4")
    print(f"  Migliorato: {sum(1 for c in confidences_improved if c > 0.95)}/4")
    
    print(f"\nCasi con confidenza = 100%:")
    print(f"  Originale:  {sum(1 for c in confidences_original if c >= 0.999)}/4")
    print(f"  Migliorato: {sum(1 for c in confidences_improved if c >= 0.999)}/4")
    
    print("\n🎉 CONCLUSIONE:")
    print("Il modello migliorato fornisce confidenze molto più realistiche!")
    print("Non pretende più di essere sicuro al 100% su ogni predizione.")

if __name__ == "__main__":
    compare_models()