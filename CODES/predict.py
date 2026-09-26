import pandas as pd
import pickle

# Your actual filenames
TRAINING_CSV = 'fetal_health.csv'
MODEL_PATH = 'FinalModel2.pkl'
SCALER_PATH = 'scaler.pkl'

# Load training feature names
train_df = pd.read_csv(TRAINING_CSV)
expected_features = train_df.drop(columns=['fetal_health'], errors='ignore').columns.tolist()

# Sample test input — update values only, not column names
new_data = pd.DataFrame([{
    col: 0 for col in expected_features  # Replace 0s with real values if needed
}])

# Load model and scaler
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Validate input columns
missing = set(expected_features) - set(new_data.columns)
extra = set(new_data.columns) - set(expected_features)

if missing:
    print(f"❌ Missing columns: {missing}")
elif extra:
    print(f"⚠️ Unexpected columns: {extra}")
else:
    scaled = scaler.transform(new_data)
    prediction = model.predict(scaled)
    proba = model.predict_proba(scaled)

    print(f"✅ Prediction: {prediction[0]}")
    print("🔍 Confidence scores:")
    for i, p in enumerate(proba[0]):
        print(f"  Class {i}: {p:.4f}")
