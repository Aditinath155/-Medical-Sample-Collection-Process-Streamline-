# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:40:13 2022

@author: user
"""
import pickle
import streamlit as st
import numpy as np

data = pickle.load(open('final.pkl','rb'))
lr_clf = pickle.load(open('train.pkl','rb'))

st.title("Medical Sample Collection Process Streamline")

#Name_Of_Test
Name_Of_Test = st.selectbox('Name_Of_Test', data["Name_Of_Test"].unique())
if Name_Of_Test == "Acute kidney profile":
    Name_Of_Test = 0
elif Name_Of_Test == "HbA1c":
    Name_Of_Test = 5
elif Name_Of_Test == "Vitamin D-25Hydroxy":
    Name_Of_Test = 9
elif Name_Of_Test == "TSH":
    Name_Of_Test = 8
elif Name_Of_Test == "Lipid Profile":
    Name_Of_Test = 6
elif Name_Of_Test == "Complete Urinalysis":
    Name_Of_Test = 2
elif Name_Of_Test == "RTPCR":
    Name_Of_Test = 7
elif Name_Of_Test == "H1N1":
    Name_Of_Test = 4
elif Name_Of_Test == "Fasting blood sugar":
    Name_Of_Test = 3
else:
    Name_Of_Test = 1

#Sample
Sample = st.radio('Sample', data["Sample"].unique())
if Sample == "Blood":
    Sample = 0
elif Sample == "Swab":
    Sample = 2
else:
    Sample = 1

#Way_Of_Storage_Of_Sample
Way_Of_Storage_Of_Sample = st.radio('Way_Of_Storage_Of_Sample', data["Way_Of_Storage_Of_Sample"].unique())
if Way_Of_Storage_Of_Sample == "Advanced":
    Way_Of_Storage_Of_Sample = 0
else:
    Way_Of_Storage_Of_Sample = 1

#Test_Booking_Time_HH_MM
Test_Booking_Time_HH_MM = st.number_input('Test_Booking_Time_HH_MM')

#Scheduled_Sample_Collection_Time_HH_MM
Scheduled_Sample_Collection_Time_HH_MM = st.number_input('Scheduled_Sample_Collection_Time_HH_MM')

#Cut_off_Schedule
Cut_off_Schedule = st.radio('Cut_off_Schedule', data['Cut-off Schedule'].unique())
if Cut_off_Schedule == "Sample by 5pm":
    Cut_off_Schedule = 1
else:
    Cut_off_Schedule = 0

#Cut_off_time_HH_MM
Cut_off_time_HH_MM = st.number_input('Cut_off_time_HH_MM')

#Agent_ID
Agent_ID = st.number_input('Agent_ID')

#Traffic_Conditions
Traffic_Conditions = st.radio('Traffic_Conditions', data['Traffic_Conditions'].unique())
if Traffic_Conditions == "Low Traffic":
    Traffic_Conditions = 1
elif Traffic_Conditions == "Medium Traffic":
    Traffic_Conditions = 2
else:
    Traffic_Conditions = 0

#Agent_Location_KM
Agent_Location_KM = st.number_input('Agent_Location_KM')

#Time_Taken_To_Reach_Patient_MM
Time_Taken_To_Reach_Patient_MM = st.number_input('Time_Taken_To_Reach_Patient_MM')

#Time_For_Sample_Collection_MM
Time_For_Sample_Collection_MM = st.number_input('Time_For_Sample_Collection_MM')

#Lab_Location_KM
Lab_Location_KM = st.number_input('Lab_Location_KM')

#Time_Taken_To_Reach_Lab_MM
Time_Taken_To_Reach_Lab_MM = st.number_input('Time_Taken_To_Reach_Lab_MM')

if st.button('Predict Result'):

    query = np.array([Name_Of_Test,Sample,Way_Of_Storage_Of_Sample,Test_Booking_Time_HH_MM,Scheduled_Sample_Collection_Time_HH_MM,Cut_off_Schedule,Cut_off_time_HH_MM,Agent_ID,Traffic_Conditions,Agent_Location_KM,Time_Taken_To_Reach_Patient_MM,Time_For_Sample_Collection_MM,Lab_Location_KM,Time_Taken_To_Reach_Lab_MM])
    query = query.reshape(1,14)

    result = lr.predict(query)

    if result == 'Y':
        st.header("Sample Reached On Time ?\n YES")
    else:
        st.header("Sample Reached On Time ?\n NO")
