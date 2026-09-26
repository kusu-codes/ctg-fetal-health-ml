# generate_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score 
from imblearn.over_sampling import SMOTE

# 1. Load Dataset
df = pd.read_csv("fetal_health.csv")

# 2. Features and Target
X = df.drop(columns=['fetal_health'])
y = df['fetal_health']

# ✅ Save feature column order
with open("feature_order.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# 3. Split first!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Apply SMOTE only to training set
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Resampled class distribution (training):")
print(y_train_res.value_counts())

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# 6. Train Model (Random Forest)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train_res)

# 7. Evaluate Model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Save Model and Scaler
with open("FinalModel2.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")
