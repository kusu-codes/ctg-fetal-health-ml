import pickle
import pandas as pd
import streamlit as st
import time

# Load model and scaler
load_model = pickle.load(open('FinalModel2.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_order = pickle.load(open('feature_order.pkl', 'rb'))
 
# Label mapping
label_map = {1: "Normal", 2: "Suspect", 3: "Pathological"}

# Helper function
def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False



# Manual Input Page
def manual_input():
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Enter Patient Data</h2>", unsafe_allow_html=True)

    inputs = {}
    cols = st.columns(2)
    for i, feature in enumerate(feature_order):
        with cols[i % 2]:
            inputs[feature] = st.text_input(feature)

    if st.button("Predict"):
        if not all(is_number(val) for val in inputs.values()):
            st.error("⚠ Please enter only numeric values for all fields.")
        else:
            input_data = [float(inputs[feature]) for feature in feature_order]
            scaled_input = scaler.transform([input_data])
            prediction = load_model.predict(scaled_input)[0]

            with st.spinner('Analyzing...'):
                time.sleep(2)

            label = label_map[prediction]
            if label == "Normal":
                st.success("✅ Normal")
            elif label == "Suspect":
                st.warning("⚠️ Suspect")
            else:
                st.error("🚨 Pathological")



# File Upload Page
def file_upload():
    st.markdown("<h2 style='text-align: center; color: #2196F3;'>Fetal Health Detection - File Upload</h2>", unsafe_allow_html=True)
    st.info("📢 *Upload a CSV file with columns in the exact order as shown below.* You can download a sample CSV template.")

    # Sample CSV download
    sample_data = pd.DataFrame([[120, 0.003, 0, 0, 0, 0, 0, 30, 10, 50, 20, 60, 60, 120, 3, 5, 100, 110, 105, 20, 0]],
                               columns=feature_order)
    csv = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Sample CSV", csv, "sample_input.csv", "text/csv")

    uploaded_file = st.file_uploader("Choose a CSV file")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("Predict"):
            try:
                X = df.reindex(columns=feature_order)
                scaled_X = scaler.transform(X)
                predictions = load_model.predict(scaled_X)
                df['Prediction'] = [label_map[p] for p in predictions]
                st.write(df[['Prediction']])
            except Exception as e:
                st.error(f"❌ Please check CSV format. Error: {e}")
# Info Page
def app_info():
    st.markdown("<h2 style='text-align: center; color: #FF5722;'>About this App</h2>", unsafe_allow_html=True)
    st.write("""
    This application predicts **fetal health** (Normal, Suspect, or Pathological) 
    using data from Cardiotocography (CTG) tests.  
    - **Manual Mode:** Enter patient CTG data manually.  
    - **File Upload Mode:** Upload a CSV file to perform batch predictions.  
    - The model used is a **Random Forest Classifier** trained with **SMOTE** to balance classes.
    """)

# Feature Details Page
def feature_details():
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Feature Information</h2>", unsafe_allow_html=True)
    st.write("""
    **Key Features Explained:**  
    - **baseline value:** Baseline fetal heart rate (bpm).  
    - **accelerations:** Number of accelerations per second.  
    - **fetal_movement:** Number of fetal movements.  
    - **uterine_contractions:** Number of uterine contractions.  
    - **light_decelerations:** Mild decelerations in heart rate.  
    - **severe_decelerations:** Severe decelerations (if present).  
    - **prolongued_decelerations:** Long decelerations.  
    - **abnormal_short_term_variability:** % of abnormal short-term variability.  
    - **mean_value_of_short_term_variability:** Average short-term variability.  
    - **percentage_of_time_with_abnormal_long_term_variability:** % of abnormal long-term variability.  
    - **histogram_*:** Various statistical parameters derived from fetal heart rate histogram.
    """)


# Main function
def main():
    st.markdown("<h1 style='text-align: center; color: #FF5722;'>Fetal Health Classification system</h1>", unsafe_allow_html=True)
    st.sidebar.title("APP MENU")
    menu = ["Manual data upload", "File data upload" ,"App Info", "Feature Details", ]
    choice = st.sidebar.radio("Select Mode", menu)

    if choice == "Manual data upload":
       manual_input() 
    elif choice == "File data upload":
        file_upload()
    elif choice == "App Info":
         app_info()
    else:
       feature_details() 

if __name__ == '__main__':
    main()