import streamlit as st
import pandas as pd
import numpy as np
from prediction import get_prediction, ordinal_encoding
import joblib

model = joblib.load("Models/xtree.pkl")

st.set_page_config(page_title="Road Traffic Accident Severity Prediction",page_icon="🚙",layout="wide")

Day_options = ['Wednesday', 'Sunday', 'Saturday', 'Tuesday', 'Monday', 'Thursday','Friday']
Driver_age_options = ['Under 18', '18-30', '31-50', 'Over 51', 'Unknown']
Sex_driver_options = ['Male', 'Female', 'Unknown']
Education_level_options = ['High school', 'Elementary school', 'Junior high school','Writing & reading', 
        'Above high school', 'Unknown','Illiterate']
Area_accident_options = ['Other', ' Church areas', 'Residential areas', 'Office areas',' Outside rural areas', 
        '  Recreational areas', ' Hospital areas','Rural village areas', ' Industrial areas', 'School areas', 
        '  Market areas', 'Rural village areasOffice areas', 'Unknown','Recreational areas']
Driver_experience = ['1-2yr', 'Below 1yr', '5-10yr', '2-5yr', 'Above 10yr',
        'No Licence', 'unknown']
Lightness_options = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting','Darkness - lights unlit']
Weather_options = ['Normal', 'Unknown', 'Raining', 'Other', 'Cloudy','Raining and Windy', 'Windy', 'Fog or mist', 'Snow']
Collision_type_options = ['Collision with roadside objects','Vehicle with vehicle collision', 'Fall from vehicles',
        'Collision with animals', 'Collision with pedestrians','Rollover', 'Other', 
        'Collision with roadside-parked vehicles','Unknown', 'With Train']
no_casualties_options = [1, 6, 2, 4, 3, 5, 7, 8]
no_vehicles_options = [2, 4, 1, 6, 3, 7]
Hour_options = [17,  1, 14, 22,  8, 15, 12, 18, 13, 20, 16, 21,  9, 10, 19, 11, 23,
        7,  0,  5,  6,  4,  3,  2]
Cause_acc_options = ['Moving Backward', 'Overtaking', 'Changing lane to the left',
        'Changing lane to the right', 'Overloading', 'Other',
        'No priority to vehicle', 'No priority to pedestrian',
        'No distancing', 'Getting off the vehicle improperly',
        'Improper parking', 'Overspeed', 'Driving carelessly',
        'Driving at high speed', 'Driving to the left', 'Unknown',
        'Overturning', 'Turnover', 'Driving under the influence of drugs',
        'Drunk driving']

features = ['Day_of_week', 'Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 'Driving_experience', 'Area_accident_occured', 'Light_conditions', 'Weather_conditions', 'Type_of_collision', 'Number_of_vehicles_involved', 'Number_of_casualties', 'Hour', 'Cause_of_accident']

st.markdown("<h1>Accident Severity Prediction</h1>", unsafe_allow_html=True)

def main():
        with st.form('prediction_form'):
                st.subheader("Enter input for following features!")
                hour = st.slider("Time (Hour) of accident:", 0, 23, value=0, format="%d")
                day_of_week = st.selectbox("Day of Week:", options=Day_options)
                driver_age = st.selectbox("Age of driver:", options=Driver_age_options)
                Sex_driver = st.selectbox("Driver Sex:", options=Sex_driver_options)
                Education_levels = st.selectbox("Education level:", options=Education_level_options)
                Accident_area = st.selectbox("Area of Accident:", options=Area_accident_options)
                Light = st.selectbox("Amount of Light:", options=Lightness_options)
                Weather = st.selectbox("Select the weather:", options =Weather_options)
                coll_type = st.selectbox("Select type of collision:", options =Collision_type_options)
                no_cas = st.slider("Number of casualties:", 0, 8, value=0, format="%d")
                no_veh = st.slider("Number of vehicles", 0, 7, value=0, format='%d')
                accident_cause = st.selectbox("Cause of accident:", options=Cause_acc_options)
                driver_exp = st.selectbox("Driver Experience:", options=Driver_experience)

                submit = st.form_submit_button("Predict")

                if submit:
                        day_of_week = ordinal_encoding(day_of_week, Day_options)
                        driver_age = ordinal_encoding(driver_age, Driver_age_options)
                        Sex_driver = ordinal_encoding(Sex_driver, Sex_driver_options)
                        Education_levels = ordinal_encoding(Education_levels, Education_level_options)
                        Accident_area = ordinal_encoding(Accident_area, Area_accident_options)
                        Light = ordinal_encoding(Light, Lightness_options)
                        Weather = ordinal_encoding(Weather, Weather_options)
                        coll_type = ordinal_encoding(coll_type, Collision_type_options)
                        accident_cause = ordinal_encoding(accident_cause, Cause_acc_options)
                        driver_exp = ordinal_encoding(driver_exp, Driver_experience)

                        data = np.array([day_of_week, driver_age, Sex_driver,
                                        Education_levels, driver_exp, Accident_area, Light,
                                        Weather, coll_type, no_veh, no_cas, hour,
                                        accident_cause]).reshape([1, -1])

                        pred = get_prediction(data=data, model=model)

                        st.write(f"The predicted accident severity is {pred[0]}")

if __name__ == '__main__':
        main()