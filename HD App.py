## 🔹Step 1: Import Necessary Libraries

import streamlit as st
import joblib
import pandas as pd
import altair as alt

## 🔹Step 2: Load preprocessor and models

preprocessor = joblib.load('preprocessor.pkl')
models = {
    "Logistic Regression": joblib.load('logistic_regression.pkl'),
    "Random Forest": joblib.load('random_forest.pkl'),
    "Decision Tree": joblib.load('decision_tree.pkl'),
    "KNN": joblib.load('knn.pkl'),
    "Naive Bayes": joblib.load('naive_bayes.pkl')
}

## 🔹Step 3: UI design

# App title
st.title("❤️ Heart Disease Risk Prediction")

# Sidebar with user input features
st.sidebar.header("Patient Information")

def user_input_features():
    age = st.sidebar.slider("Age", 20, 110, 50)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", 
                             ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
    trestbps = st.sidebar.slider("Resting Blood Pressure (systolic) (mm Hg)", 70, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.sidebar.selectbox("Resting ECG Results", 
                                 ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 70, 270, 170)
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    
    # Convert the inputs from the user into model input format
    data = {
        'age': age,
        'sex': 1 if sex == "Male" else 0,
        'cp': ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp),
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == "Yes" else 0,
        'restecg': ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg),
        'thalach': thalach,
        'exang': 1 if exang == "Yes" else 0,
        'oldpeak': oldpeak
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Display user input
st.subheader("Patient Details")
st.write(input_df)

# Preprocess and predict
if st.button("Predict"):
    # Preprocess
    processed_data = preprocessor.transform(input_df)
    
    # Display predictions
    st.subheader("Prediction Results")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Predictions**")
        predictions = {}
        for name, model in models.items():
            pred = model.predict(processed_data)[0]
            predictions[name] = "Positive" if pred == 1 else "Negative"
            st.write(f"{name}: {predictions[name]}")
    
    with col2:
        st.markdown("**Risk Probabilities**")
        probabilities = {}
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(processed_data)[0][1]
                probabilities[name] = proba
                st.write(f"{name}: {proba:.1%}")
    
    # Probability Chart
    st.subheader("Risk Probability Comparison")

    if probabilities:  # Ensure probabilities exist
        prob_df = pd.DataFrame.from_dict(probabilities, orient='index', columns=['Probability']).reset_index()
        prob_df.rename(columns={'index': 'Model'}, inplace=True)

        chart = alt.Chart(prob_df).mark_bar().encode(
            x="Probability:Q",
            y=alt.Y("Model:N", sort='-x'),  # Sort by probability in descending order
            color=alt.Color("Probability:Q", scale=alt.Scale(domain=[0, 1], range=["green", "red"]), legend=None),
            tooltip=["Model", alt.Tooltip("Probability:Q", format=".1%")]
        ).properties(width=500, height=300).mark_bar(size=30)  # Adjust bar size for better readability

        text = chart.mark_text(
            align='left',
            baseline='middle',
            dx=3  # Move text slightly right of bars
        ).encode(
            text=alt.Text("Probability:Q", format=".1%")
        )

        st.altair_chart(chart + text, use_container_width=True)  # Combine bars & labels
    
    # Interpretation
    st.subheader("Interpretation")
    st.write("""
    - **Positive**: Indicates higher risk of heart disease
    - **Negative**: Indicates lower risk of heart disease
    - Probability shows the confidence level of the prediction
    """)

## 🔹Step 4: Information about the app

st.sidebar.markdown("""
**About This App**  
This app predicts the risk of heart disease using multiple machine learning models.  
The models were trained on clinical parameters from patients.
""")

## 🔹Step 5: Disclaimer

st.subheader("⚠️ Disclaimer")
st.write("""
This application is for **informational and educational purposes only**.  
It does **not** provide medical advice, diagnosis, or treatment.  
Always consult with a **qualified healthcare professional** for medical concerns.  
The predictions are based on machine learning models and may not be 100% accurate.
""")
