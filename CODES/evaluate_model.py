import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_csv("fetal_health.csv")
X = df.drop(columns=['fetal_health'])
y_true = df['fetal_health']

# 2. Load Model and Scaler
with open("FinalModel2.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 3. Scale Features
X_scaled = scaler.transform(X)

# 4. Predict
y_pred = model.predict(X_scaled)

# 5. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# 6. Class Prediction Info
unique_classes = np.unique(y_pred)
print(f"✅ Model is predicting across {len(unique_classes)} classes: {unique_classes}")

# 7. Accuracy
accuracy = np.mean(y_true == y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
if accuracy < 0.70:
    print("⚠️ Low accuracy. Model may need retraining or data issue check.")
else:
    print("✅ Model performs well on the full dataset.")

# 8. Save Confusion Matrix Plot
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("📸 Confusion matrix saved as 'confusion_matrix.png'")

# 9. Save Classification Report Heatmap
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Drop 'accuracy' row if present
if "accuracy" in report_df.index:
    report_df = report_df.drop(index=["accuracy"])

plt.figure(figsize=(8, 4))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Classification Report")
plt.tight_layout()
plt.savefig("evaluation_report.png")
plt.close()
print("📊 Evaluation report saved as 'evaluation_report.png'")
