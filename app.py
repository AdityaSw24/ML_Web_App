import numpy as np
import pickle
import streamlit as st

# Model loaded
loaded_classifier=pickle.load(open("trained_classifier.sav","rb"))

# Prediction Function

def diabetes_prediction(input_data):

    # input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape as single row data as  [12,4,1...... ]
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_classifier.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
def main():

    st.title('Diabetes Predictor')

    pregnancies = st.text_input("Number of pregnancies")
    glucose = st.text_input("Glucose level")
    blood_pressure = st.text_input("Blood Pressure value")
    skin_thickness = st.text_input("Skin Thickness value")
    insulin = st.text_input("Insulin level")
    bmi = st.text_input("BMI value")
    dpf = st.text_input("Diabetes Pedigree Function value")
    age = st.text_input("Age of the Person")

    diagnosis=''

    # button for prediction

    if st.button("Test Result"):
        diagnosis= diabetes_prediction([pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,dpf,age])
    
    st.success(diagnosis)

if __name__=='__main__':
    main()