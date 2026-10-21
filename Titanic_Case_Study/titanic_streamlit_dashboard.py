import streamlit as st
import joblib as jb
import pandas as pd

st.set_page_config(page_title = "Titanic Ship Incident ML Dashboard")

st.markdown(
    "<h2 style='text-align: center; color: #25627a;'>Titanic Ship Incident Death Prediction</h2>", 
    unsafe_allow_html = True
)

trained_model = jb.load("titanic_cs.pkl")

passenger_class = st.selectbox(
    "Select Passenger Travelling Class.",
    ("1st Class", "2nd Class", "3rd Class"),
    index = None,
    placeholder = "Select passenger class...",
)

passenger_gender = st.selectbox(
    "Select Passenger Gender.",
    ("Male", "Female"),
    index = None,
    placeholder = "Select passenger gender...",
)

passenger_age = st.number_input(
    "Enter Passanger Age",
    min_value = 1,
    value = None
)

siblings_spouse_count = st.number_input(
    "Enter Siblings / Spouse aboard along with the passanger",
    min_value = 0,
    value = None
)

parents_children_count = st.number_input(
    "Enter Parents / Children aboard along with the passanger",
    min_value = 0,
    value = None
)

ship_fare = st.number_input(
    "Enter ship fare",
    value = None,
    format="%.2f",
    icon="💲"
)

passenger_boarding_point = st.selectbox(
    "Select Passenger's Boarding Point.",
    ("Cherbourg", "Queenstown", "Southampton"),
    index = None,
    placeholder = "Select passenger's boarding point...",
)

btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

required_values = (
    passenger_class,
    passenger_gender,
    passenger_age,
    siblings_spouse_count,
    parents_children_count,
    ship_fare,
    passenger_boarding_point,
)
all_fields_filled = all(value is not None for value in required_values)

if btn_col2.button(
    "Predict Survival Chances",
    use_container_width = True,
    disabled = not all_fields_filled
):
    if not all_fields_filled:
        st.error("Please complete all fields before predicting.")
        st.stop()

    if passenger_class == "1st Class":
        p_class = 1
    elif passenger_class == "2nd Class":
        p_class = 2
    else:
        p_class = 3

    passanger = pd.DataFrame([{
        "Pclass": p_class,
        "Sex": 1 if passenger_gender == "Female" else 0,
        "Age": passenger_age,
        "sibsp": siblings_spouse_count,
        "Parch": parents_children_count,
        "Fare": ship_fare,
        "Embarked_1.0": 1 if passenger_boarding_point == "Queenstown" else 0,
        "Embarked_2.0": 1 if passenger_boarding_point == "Southampton" else 0
    }])

    passanger = passanger[trained_model.feature_names_in_]

    result = trained_model.predict(passanger)[0]
    
    if result == 1:
        st.markdown(
            f"<h4 style='text-align: center; color: green;'>The passenger will survive</h4>", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h4 style='text-align: center; color: red;'>The passenger won't survive</h4>", 
            unsafe_allow_html=True
        )
