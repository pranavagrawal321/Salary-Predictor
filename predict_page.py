import streamlit as st
import pickle
import numpy as np

st.set_page_config(layout="wide")


def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


def show_predict_page():
    st.title("Software Developer Salary Prediction")
    
    countries = (
        "United States of America",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )
    
    education = (
        "Less",
        "Bachelors",
        "Masters",
        "Postgrad",
    )
    
    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)
    
    expericence = st.slider("Years of Experience", 0, 50, 3)
    
    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X = X.astype(float)
        
        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
