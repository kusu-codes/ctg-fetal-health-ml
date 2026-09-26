# Fetal Health Classification

This project uses machine learning to classify fetal health conditions into **Normal**, **Suspect**, and **Pathological** based on Cardiotocography (CTG) data. It aims to assist in early diagnosis and decision-making in prenatal care.

---

├── app.py               # Streamlit web app for predictions  
├── generate_model.py    # Model training and saving code  
├── fetal_health.csv     # Dataset from UCI repository  
├── finalmodel2.pkl      # Trained Random Forest model  
├── scaler.pkl           # StandardScaler object used for feature scaling  
├── feature_order.pkl    # Saved list of feature column names  
├── requirements.txt     # List of dependencies  
└── README.md            # Project documentation  

## ⚙️ Technologies Used

- Python 3.x
- pandas, numpy
- scikit-learn
- imbalanced-learn (for SMOTE)
- matplotlib, seaborn
- Streamlit (for UI)

## 📌 Dataset

- **Source**: [Kaggle – Fetal Health Classification Dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)
- **Description**: The dataset includes 21 features derived from Cardiotocography (CTG) readings, such as fetal heart rate and uterine contractions.
- **Target Classes**:
  - 1 → Normal
  - 2 → Suspect
  - 3 → Pathological


---

## 🧪 Steps Performed

1. **Data Cleaning and Preprocessing**
2. **Train-Test Split (80:20 ratio)**
3. **Applied SMOTE** to balance class distribution
4. **Model Training** using Random Forest Classifier
5. **Evaluation** using accuracy, precision, recall, F1-score
6. **Deployed** using a simple Streamlit web app

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```
  
Launch the App
-m streamlit run app.py
