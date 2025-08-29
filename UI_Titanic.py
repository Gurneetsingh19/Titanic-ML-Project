import pandas as pd
import streamlit as st
import pybase64
import joblib
import numpy as np

st.set_page_config(layout="centered",page_title="Titanic Prediction Model",page_icon="🚢")

model = joblib.load("Model_traning/titanic.joblib") # load Trained model
scaler = joblib.load("Model_traning/titanic_scaler.joblib") # load scaler

st.title("Titanic Survival Predictor 🚢")

def img_to_base64(img): #Background img
    with open(img,"rb") as f:
        return pybase64.b64encode(f.read()).decode("utf-8") 
    
imgg=img_to_base64("UI/backimg.png") 
# Style background image.
bg_img=f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{imgg}");
        background-size: cover;
    }}
    </style>
'''
st.markdown(  # Form background 
    """
    <style>
    /* Form background color */
    .stForm {
        background: rgba(255, 255, 255, 0.2);  /* white with 20% pacity */
        backdrop-filter: blur(10px);            /* blur effect */
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);  /* optional shadow */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(bg_img,unsafe_allow_html=True)

#Form 

with st.form("Form"):
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 0, 100, 25)
    sibsp = st.number_input("Number of siblings/spouses", 0, 10, 0)
    parch = st.number_input("Number of parents/children", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 600.0,50.0)
    st.write("Enter the Titanic ticket price in Pounds (e.g., 7.25, 71.28, 30.07)")
    embarked = st.selectbox("Embarked", ["C : Cherbourg (France)",
        "Q : Queenstown (Ireland)",
        "S: Southampton (England)"])
    btn = st.form_submit_button("Submit")


#Encoding
sex = 1 if sex=="Male" else 0
embarked_c = 1 if embarked == "C" else 0
embarked_q = 1 if embarked == "Q" else 0

#scalling 
scaled_value = scaler.transform([[age,fare]])
age_scaled = scaled_value[0][0]
fare_scaled = scaled_value[0][1]

X = np.array([[pclass, sex, age_scaled, sibsp, parch, fare_scaled, embarked_c, embarked_q]])


if btn:
    prediction = model.predict(X)
    if prediction[0] == 1:
        st.markdown(
        """
        <div style="
            background-color: rgba(0, 128, 0, 1);  
            color: white;
            padding: 15px;
            border-radius: 10px;
            font-weight: bold;
            text-align: center;
        ">
            Congratulation! you would survived.
        </div>
        """,
        unsafe_allow_html=True
        )
    else:
        st.markdown(
        """
        <div style="
            background-color: rgba(255, 0, 0, 1);  
            color: white;
            padding: 15px;
            border-radius: 10px;
            font-weight: bold;
            text-align: center;
        ">
            😢 Sorry, you would NOT have survived.
        </div>
        """,
        unsafe_allow_html=True
        )