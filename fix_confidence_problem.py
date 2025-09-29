# SOLUZIONI PER RISOLVERE IL PROBLEMA DELLA CONFIDENZA AL 100%

## 1. RIADDESTRARE IL MODELLO CON PARAMETRI PIÙ CONSERVATIVI

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def create_better_model():
    """Crea un modello Random Forest con parametri più conservativi"""
    
    # Carica il dataset originale
    df = pd.read_csv('student_pe_performance.csv')
    
    # Prepara le features (escludi ID e Performance)
    features = df.drop(['ID', 'Performance'], axis=1)
    target = df['Performance']
    
    # Label encoding per variabili categoriche
    label_encoders = {}
    categorical_columns = ['Gender', 'Grade_Level', 'Class_Participation_Level', 
                          'Motivation_Level', 'Final_Grade', 'Previous_Semester_PE_Grade']
    
    for col in categorical_columns:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])
        label_encoders[col] = le
    
    # Label encoder per il target
    target_le = LabelEncoder()
    target_encoded = target_le.fit_transform(target)
    label_encoders['Performance'] = target_le
    
    # Standardizzazione
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split del dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target_encoded, test_size=0.2, random_state=42, stratify=target_encoded
    )
    
    # MODELLO MIGLIORATO CON PARAMETRI CONSERVATIVI
    rf_improved = RandomForestClassifier(
        n_estimators=50,           # Ridotto da 300 a 50
        max_depth=8,               # Ridotto da 15 a 8
        min_samples_split=10,      # Aumentato da 2 a 10
        min_samples_leaf=5,        # Aumentato da 1 a 5
        max_features='sqrt',       # Invece di None, usa sqrt
        bootstrap=True,
        random_state=42,
        class_weight='balanced'    # Bilancia le classi
    )
    
    # Addestramento
    rf_improved.fit(X_train, y_train)
    
    # Validazione incrociata
    cv_scores = cross_val_score(rf_improved, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Test
    y_pred = rf_improved.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_le.classes_))
    
    # Testa le probabilità su campioni del test set
    print("\n=== TEST CONFIDENZE SU CAMPIONI REALI ===")
    test_probabilities = rf_improved.predict_proba(X_test)
    max_confidences = np.max(test_probabilities, axis=1)
    
    print(f"Confidenza media: {np.mean(max_confidences):.4f}")
    print(f"Confidenza massima: {np.max(max_confidences):.4f}")
    print(f"Confidenza minima: {np.min(max_confidences):.4f}")
    print(f"Campioni con confidenza > 95%: {np.sum(max_confidences > 0.95)}/{len(max_confidences)}")
    print(f"Campioni con confidenza = 100%: {np.sum(max_confidences == 1.0)}/{len(max_confidences)}")
    
    # Salva i modelli migliorati
    joblib.dump(rf_improved, 'random_forest_model_improved.pkl')
    joblib.dump(scaler, 'scaler_improved.pkl')
    joblib.dump(label_encoders, 'label_encoders_improved.pkl')
    
    print("\n✅ Modelli migliorati salvati con suffisso '_improved'")
    
    return rf_improved, scaler, label_encoders

## 2. ALTERNATIVE AL RANDOM FOREST

def create_alternative_models():
    """Crea modelli alternativi che potrebbero dare confidenze più realistiche"""
    
    # Carica il dataset
    df = pd.read_csv('student_pe_performance.csv')
    features = df.drop(['ID', 'Performance'], axis=1)
    target = df['Performance']
    
    # Preprocessing (stesso del Random Forest)
    label_encoders = {}
    categorical_columns = ['Gender', 'Grade_Level', 'Class_Participation_Level', 
                          'Motivation_Level', 'Final_Grade', 'Previous_Semester_PE_Grade']
    
    for col in categorical_columns:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])
        label_encoders[col] = le
    
    target_le = LabelEncoder()
    target_encoded = target_le.fit_transform(target)
    label_encoders['Performance'] = target_le
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target_encoded, test_size=0.2, random_state=42, stratify=target_encoded
    )
    
    # MODELLO 1: Gradient Boosting (spesso più calibrato)
    from sklearn.ensemble import GradientBoostingClassifier
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    # MODELLO 2: SVM con probabilità
    from sklearn.svm import SVC
    svm_model = SVC(
        kernel='rbf',
        probability=True,  # Importante per le probabilità
        random_state=42
    )
    svm_model.fit(X_train, y_train)
    
    # MODELLO 3: Logistic Regression (naturalmente calibrato)
    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    
    # Test delle confidenze per tutti i modelli
    models = {
        'Gradient Boosting': gb_model,
        'SVM': svm_model,
        'Logistic Regression': lr_model
    }
    
    print("\n=== CONFRONTO CONFIDENZE TRA MODELLI ===")
    for name, model in models.items():
        test_probabilities = model.predict_proba(X_test)
        max_confidences = np.max(test_probabilities, axis=1)
        
        print(f"\n{name}:")
        print(f"  Confidenza media: {np.mean(max_confidences):.4f}")
        print(f"  Confidenza massima: {np.max(max_confidences):.4f}")
        print(f"  Campioni con confidenza > 95%: {np.sum(max_confidences > 0.95)}/{len(max_confidences)}")
        print(f"  Campioni con confidenza = 100%: {np.sum(max_confidences == 1.0)}/{len(max_confidences)}")

## 3. CALIBRAZIONE DELLE PROBABILITÀ

def calibrate_model():
    """Calibra le probabilità del modello esistente"""
    from sklearn.calibration import CalibratedClassifierCV
    
    # Carica il modello esistente
    rf_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    
    # Carica il dataset per ri-calibrare
    df = pd.read_csv('student_pe_performance.csv')
    features = df.drop(['ID', 'Performance'], axis=1)
    target = df['Performance']
    
    # Preprocessing
    for col in ['Gender', 'Grade_Level', 'Class_Participation_Level', 
                'Motivation_Level', 'Final_Grade', 'Previous_Semester_PE_Grade']:
        if col in label_encoders and col in features.columns:
            le = label_encoders[col]
            features[col] = features[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            features[col] = le.transform(features[col])
    
    features_scaled = scaler.transform(features)
    target_encoded = label_encoders['Performance'].transform(target)
    
    # Calibrazione del modello
    calibrated_rf = CalibratedClassifierCV(rf_model, method='isotonic', cv=3)
    calibrated_rf.fit(features_scaled, target_encoded)
    
    # Salva il modello calibrato
    joblib.dump(calibrated_rf, 'random_forest_model_calibrated.pkl')
    
    print("✅ Modello calibrato salvato come 'random_forest_model_calibrated.pkl'")
    
    return calibrated_rf

if __name__ == "__main__":
    print("🔧 CREAZIONE MODELLO MIGLIORATO...")
    create_better_model()
    
    print("\n🔧 CREAZIONE MODELLI ALTERNATIVI...")
    create_alternative_models()
    
    print("\n🔧 CALIBRAZIONE MODELLO ESISTENTE...")
    calibrate_model()