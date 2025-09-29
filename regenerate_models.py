#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

print("🔄 Rigenerando i modelli con le nuove versioni...")

# Carica il dataset
data = pd.read_csv("student_pe_performance.csv")
print(f"📊 Dataset caricato: {data.shape[0]} righe, {data.shape[1]} colonne")

# Rimuovi ID se presente
if 'ID' in data.columns:
    data = data.drop('ID', axis=1)
    print("✅ Colonna ID rimossa")

# Gestione dati mancanti
data = data.dropna()

# Label encoding per le variabili categoriche
categorical_cols = data.select_dtypes(include=['object']).columns
print(f"📋 Colonne categoriche trovate: {list(categorical_cols)}")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
    print(f"✅ {col}: {len(le.classes_)} classi - {list(le.classes_)}")

# Divisione feature/target
X = data.drop("Performance", axis=1)
y = data["Performance"]

print(f"\n📊 Dimensioni finali:")
print(f"   Features (X): {X.shape}")
print(f"   Target (y): {y.shape}")

# Standardizzazione
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🎯 Divisione train/test:")
print(f"   Training set: {X_train.shape[0]} campioni")
print(f"   Test set: {X_test.shape[0]} campioni")

# Addestra il modello con i parametri ottimali
print("\n🚀 Addestrando Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Valutazione
from sklearn.metrics import accuracy_score
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
confidence = np.mean(np.max(y_pred_proba, axis=1))

print(f"\n📈 RISULTATI:")
print(f"   🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"   🔥 Confidenza media: {confidence:.4f} ({confidence*100:.1f}%)")

# Salva i modelli
print("\n💾 Salvando i modelli...")
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# Test di caricamento
print("\n🧪 Test di caricamento...")
test_model = joblib.load("random_forest_model.pkl")
test_scaler = joblib.load("scaler.pkl")
test_encoders = joblib.load("label_encoders.pkl")

print("✅ Modelli rigenerati con successo!")
print(f"   scikit-learn version: {joblib.__version__}")

# Test con dati simulati della webapp
webapp_test_data = {
    'Age': 16, 'Gender': 1, 'Grade_Level': 2, 'Strength_Score': 65.0,
    'Endurance_Score': 70.0, 'Flexibility_Score': 60.0, 'Speed_Agility_Score': 55.0,
    'BMI': 22.5, 'Health_Fitness_Knowledge_Score': 75.0, 'Skills_Score': 68.0,
    'Class_Participation_Level': 2, 'Attendance_Rate': 85.0, 'Motivation_Level': 2,
    'Overall_PE_Performance_Score': 72.0, 'Improvement_Rate': 4.5, 'Final_Grade': 1,
    'Previous_Semester_PE_Grade': 1, 'Hours_Physical_Activity_Per_Week': 6.0
}

test_df = pd.DataFrame([list(webapp_test_data.values())], 
                      columns=list(webapp_test_data.keys()))
test_scaled = test_scaler.transform(test_df)
test_pred = test_model.predict(test_scaled)
test_proba = test_model.predict_proba(test_scaled)

print(f"\n🧪 TEST SIMULAZIONE WEBAPP:")
print(f"   🎯 Predizione: {test_pred[0]}")
print(f"   🔥 Confidenza: {np.max(test_proba[0]):.3f} ({np.max(test_proba[0])*100:.1f}%)")

if 'Performance' in test_encoders:
    result_label = test_encoders['Performance'].inverse_transform(test_pred)[0]
    print(f"   📝 Risultato: {result_label}")

print("\n🎉 TUTTO PRONTO! Modelli compatibili generati.")