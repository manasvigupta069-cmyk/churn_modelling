import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
st.title("Customer Churn Prediction")
CS= st.number_input("Credit Score")
Age= st.number_input("Age")
Tenure= st.number_input("Tenure")
Balance= st.number_input("Balance")
NumOfProducts= st.number_input("Number of Products")
Geography= st.selectbox("Geography",("France","Germany","Spain"))
Gender= st.selectbox("Gender",("Male","Female"))
Has_credit_card= st.selectbox("Has Credit Card",("Yes","No"))
Is_active_member= st.selectbox("Is Active Member",("Yes","No"))
model = tf.keras.models.load_model("churn_class.h5")

geo = {"France": 0, "Germany": 1, "Spain": 2}
gender = {"Male": 0, "Female": 1}
yes_no = {"Yes": 1, "No": 0}

if st.button("Predict"):
    input_data = np.array([[
        CS,
        Age,
        Tenure,
        Balance,
        NumOfProducts,
        geo[Geography],
        gender[Gender],
        yes_no[Is_active_member]
    ]])

    prediction = model.predict(input_data)

    if prediction[0][0] > 0.5:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is likely to Stay")
