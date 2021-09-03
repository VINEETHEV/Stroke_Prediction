

from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('Stroke_Prediction_Model')


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Describe features", "Upload csv"))
    st.sidebar.info('This app is created for stroke prediction using several features')
    st.title("Stroke Prediction")
    if add_selectbox == 'Describe features':
        gender=st.number_input('gender(Male:1,Female:0)' , min_value=0, max_value=1, value=1)
        age =st.number_input('age',min_value=0, max_value=82, value=1)
        hypertension = st.number_input('hypertension', min_value=0, max_value=1, value=1)
        heart_disease = st.number_input('heart_disease', min_value=0, max_value=1, value=1)
        ever_married = st.number_input('ever_married( Yes:1, NO:0 )',  min_value=0, max_value=1, value=1)
        Residence_type = st.number_input('Residence_type( Urban:1, Rural:0 )',  min_value=0, max_value=1, value=1)
        avg_glucose_level = st.number_input('avg_glucose_level', min_value=55, max_value=300, value=100)
        bmi = st.number_input('bmi', min_value=10, max_value=100, value=50)
        smoking_status = st.selectbox('smoking_status', ['never smoked', 'formerly smoked', 'smokes'])
        work_type = st.selectbox('work_type', ['Private', 'Self-employed','children','Govt_job','Never_worked'])
        output=""
        input_dict={'gender':gender,'age':age,'hypertension':hypertension,'heart_disease':heart_disease,'ever_married':ever_married,'work_type':work_type,'Residence_type':Residence_type, 'avg_glucose_level':avg_glucose_level, 'bmi':bmi, 'smoking_status':smoking_status}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('The output is {}'.format(output))
    if add_selectbox == 'Upload csv':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)            
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
def main():
    run()

if __name__ == "__main__":
  main()
