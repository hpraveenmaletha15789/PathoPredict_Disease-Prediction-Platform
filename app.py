import pickle
import os          # <- this is required for os.path.join
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from streamlit_option_menu import option_menu
import google.generativeai as genai

#  Gemini API Setup
genai.configure(api_key="AIzaSyDKIRO4yqXCoBqK8WXu24F6wGuu7S_GHf4")
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="You are a professional medical assistant. Provide accurate information about diseases, symptoms, and health advice. Always include a disclaimer that this is not a substitute for professional medical advice."
)

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

                   
working_dir = r"C:\Users\praveen maletha\Documents\Smart-Diagnosis-Hub-main"
# Load saved models
diabetes_model = pickle.load(open(os.path.join(working_dir, "saved_models", "diabetes_model.sav"), "rb"))
heart_disease_model = pickle.load(open(os.path.join(working_dir, "saved_models", "heart_disease_model.sav"), "rb"))
parkinsons_model = pickle.load(open(os.path.join(working_dir, "saved_models", "parkinsons_model.sav"), "rb"))



# Sidebar for navigation
with st.sidebar:
    st.markdown(
        """
        <h1><b>🏥 Multiple Disease Prediction System
        </h1>
        <hr style="border: 4px solid #007bff">
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
        menu_title="Navigation",  # Required
        options=[
            'Diabetes Prediction',
            'Heart Disease Prediction',
            'Parkinsons Prediction',
            'Health Assistant Chatbot'
        ],  # Required
        icons=['activity', 'heart', 'person', 'robot'],  # Optional
        menu_icon="hospital-fill",  # Optional
        default_index=0,  # Optional
        styles={
            
            "nav-link": {
                "font-size": "16px",
                "font-weight": "bold",
                "background-color": "#3a3f44",
                "color": "white",
                "text-align": "left",
                "margin": "5px",
            },
            "nav-link-selected": {"background-color": "#007bff", "color": "white"},
        },
    )

# Function to display a bar chart for statistical analysis
def show_bar_chart(input_data, feature_names, title):
    plt.figure(figsize=(10, 5))
    plt.bar(feature_names, input_data, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Values")
    st.pyplot(plt)

# Comparative analysis function with graph
def show_comparative_analysis(user_data, benchmark_data, feature_names, title):
    comparison = pd.DataFrame({
        'Feature': feature_names,
        'User Input': user_data,
        'Benchmark': benchmark_data
    })
    st.write(title)
    st.dataframe(comparison)

    # Display comparative bar chart
    x = np.arange(len(feature_names))  # Feature labels
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, user_data, width, label='User Input', color='skyblue')
    ax.bar(x + width/2, benchmark_data, width, label='Benchmark', color='salmon')

    ax.set_xlabel('Features')
    ax.set_ylabel('Values')
    ax.set_title('Comparative Analysis: User Input vs Benchmark')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')

    st.pyplot(fig)

# Display risks and measures
def show_risks_and_measures(disease_name):
    risks_measures = {
        "Diabetes": {
            "risks": [
                "High blood sugar can damage organs over time.",
                "Increases risk of heart disease, stroke, and kidney problems.",
                "Can lead to nerve damage and vision problems."
            ],
            "measures": [
                "Maintain a healthy diet rich in vegetables and low in processed foods.",
                "Exercise regularly to control weight and improve insulin sensitivity.",
                "Monitor blood sugar levels and avoid excessive sugar intake."
            ]
        },
        "Heart Disease": {
            "risks": [
                "Can lead to heart attacks or strokes.",
                "Increases risk of high blood pressure and cholesterol.",
                "May cause heart failure or arrhythmias over time."
            ],
            "measures": [
                "Adopt a heart-healthy diet, including fruits, vegetables, and lean proteins.",
                "Engage in regular physical activity and avoid smoking.",
                "Manage stress and monitor blood pressure and cholesterol levels."
            ]
        },
        "Parkinson's Disease": {
            "risks": [
                "Affects motor skills and coordination, leading to tremors.",
                "Progressive condition impacting quality of life over time.",
                "Can result in cognitive decline and mood disorders."
            ],
            "measures": [
                "Maintain an active lifestyle to improve muscle strength.",
                "Include omega-3 fatty acids and antioxidants in your diet.",
                "Engage in cognitive activities and seek early diagnosis for treatment."
            ]
        }
    }

    st.subheader(f"Risks and Measures for {disease_name}") 
    st.write("### Risks:")
    for risk in risks_measures[disease_name]["risks"]:
        st.write(f"- {risk}")
    st.write("### Preventive Measures:")
    for measure in risks_measures[disease_name]["measures"]:
        st.write(f"- {measure}")

# Download health report function
def download_report(disease_name, diagnosis, user_input, feature_names):
    report = f"""
    **Health Report for {disease_name}**
    
    **Diagnosis:**
    {diagnosis}
    
    **Input Features:**
    {dict(zip(feature_names, user_input))}
    
    **Recommendations:**
    Based on the results of your test, it is recommended to follow a healthy lifestyle and consult with a healthcare professional for further guidance.
    """
    
    with open(f"{disease_name}_health_report.txt", "w") as file:
        file.write(report)
    
    st.download_button(
        label="Download Health Report",
        data=open(f"{disease_name}_health_report.txt", "rb").read(),
        file_name=f"{disease_name}_health_report.txt",
        mime="text/plain"
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    # Prediction and analysis
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])
        diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
        st.success(diagnosis)

        # Statistical analysis
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age']
        show_bar_chart(user_input, feature_names, "Input Features for Diabetes Prediction")

        # Comparative analysis with benchmark
        benchmark_data = [3, 120, 80, 20, 80, 28, 0.5, 45]  # Example benchmark data
        show_comparative_analysis(user_input, benchmark_data, feature_names, "Comparative Analysis for Diabetes")

        # Show risks and measures
        show_risks_and_measures("Diabetes")
        
        # Download health report
        download_report("Diabetes", diagnosis, user_input, feature_names)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # Prediction and analysis
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, xang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        heart_prediction = heart_disease_model.predict([user_input])
        diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'
        st.success(diagnosis)

        # Statistical analysis
        feature_names = ['Age', 'Sex', 'CP', 'RestBP', 'Chol', 'FBS', 'RestECG', 'Thalach', 'Exang', 'Oldpeak', 'Slope', 'CA', 'Thal']
        show_bar_chart(user_input, feature_names, "Input Features for Heart Disease Prediction")

        # Comparative analysis with benchmark
        benchmark_data = [60, 1, 3, 130, 250, 0, 1, 150, 1, 2.0, 2, 1, 1]  # Example benchmark data
        show_comparative_analysis(user_input, benchmark_data, feature_names, "Comparative Analysis for Heart Disease")

        # Show risks and measures
        show_risks_and_measures("Heart Disease")
        
        # Download health report
        download_report("Heart Disease", diagnosis, user_input, feature_names)

# Parkinsons Prediction Page
if selected == 'Parkinsons Prediction':
    st.title('Parkinsons Prediction using ML')

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        mean_freq = st.text_input('Mean Frequency')
    with col2:
        mean_pitch = st.text_input('Mean Pitch')
    with col3:
        sd_freq = st.text_input('Standard Deviation of Frequency')
    with col1:
        skew_freq = st.text_input('Skewness of Frequency')
    with col2:
        kurt_freq = st.text_input('Kurtosis of Frequency')
    with col3:
        mean_rolloff = st.text_input('Mean Spectral Rolloff')
    with col1:
        sd_rolloff = st.text_input('Standard Deviation of Spectral Rolloff')
    with col2:
        mean_bw = st.text_input('Mean Spectral Bandwidth')

    # Prediction and analysis
    if st.button('Parkinsons Test Result'):
        user_input = [mean_freq, mean_pitch, sd_freq, skew_freq, kurt_freq, mean_rolloff, sd_rolloff, mean_bw]
        user_input = [float(x) for x in user_input]
        parkinsons_prediction = parkinsons_model.predict([user_input])
        diagnosis = 'The person has Parkinson\'s' if parkinsons_prediction[0] == 1 else 'The person does not have Parkinson\'s'
        st.success(diagnosis)

        # Statistical analysis
        feature_names = ['Mean Frequency', 'Mean Pitch', 'SD Frequency', 'Skew Frequency', 'Kurt Frequency', 'Mean Rolloff', 'SD Rolloff', 'Mean BW']
        show_bar_chart(user_input, feature_names, "Input Features for Parkinson's Prediction")

        # Comparative analysis with benchmark
        benchmark_data = [0.03, 150, 0.05, -0.2, 1.5, 0.5, 0.05, 0.25]  # Example benchmark data
        show_comparative_analysis(user_input, benchmark_data, feature_names, "Comparative Analysis for Parkinson's")

        # Show risks and measures
        show_risks_and_measures("Parkinson's Disease")
        
        # Download health report
        download_report("Parkinson's", diagnosis, user_input, feature_names)

# --- Health Assistant Chatbot Logic ---
if selected == 'Health Assistant Chatbot':
    st.title("Health Assistant Chatbot 🤖")
    st.markdown("Ask me anything about any disease, health conditions, or symptoms.")

    # Initialize session state for chatbot history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User chat input
    if user_input := st.chat_input("Type your health-related query here..."):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display Gemini response
        with st.chat_message("assistant"):
            try:
                response = model.generate_content(user_input)
                ai_message = response.text
                st.markdown(ai_message)
                # Save assistant message to state
                st.session_state.messages.append({"role": "assistant", "content": ai_message})
            except Exception as e:
                st.error(f"Error connecting to AI: {e}")
