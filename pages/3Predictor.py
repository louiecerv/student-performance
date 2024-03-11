#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Define the Streamlit app
def app():

    if "gender" not in st.session_state:
        st.session_state.gender = ''
    if "yearlevel" not in st.session_state:
        st.session_state.yearlevel = ''
    if "course" not in st.session_state:
        st.session_state.course = ''
    if "income" not in st.session_state:
        st.session_state.income = ''
    if "user_inputs" not in st.session_state:
        st.session_state.user_inputs = ''

    st.write("""The trained model will predict the E-Banking Usage Level from the information provided.""")
    
    st.subheader('User Information')
    update_selections()

    if button("Start")
        st.write('Unencoded user inputs:')
        user_inputs = np.array(st.session_state['user_inputs'])
        st.write(user_inputs)

        st.write('Encoded user inputs:')
        le_list = st.session_state["le_list"]
        encoded = []
        i = 0
        for value in user_inputs[0]:
            result = le_list[i].transform([value])
            encoded.append(result)
            i = i + 1
        encoded = pd.DataFrame(encoded)
        st.write(encoded.transpose())
        predicted =  st.session_state["clf"].predict(encoded.transpose())
        level = ''
        if predicted==0:
            level = 'Very Low'
        elif predicted==1:
            level = 'Low'
        elif predicted==2:
            level = 'Moderate'
        elif predicted==3:
            level = 'High'
        else:        
            level = 'Very High'

        result = 'The predicted Usage Level: ' + level
        st.subheader(result)
    
def update_selections():
    gender = st.selectbox('Gender:', ['Male', 'Female'])
    st.session_state['gender'] = gender
    yearlevel =  st.selectbox('Year Level: ', ['First Year', 'Second Year', 'Third Year', 'Fourth Year'])
    st.session_state['yearlevel'] = yearlevel
    course =  st.selectbox('Course: ', ['BSTM', 'BSCM', 'BSBA', 'BSHM'])
    st.session_state['course'] = course
    income =  st.selectbox('Family Income: ', ['Php 20 000 and Below', 'Above Php 60 000', 'Php 20 001 to Php 60 000'])
    st.session_state['income'] = income
    
    st.session_state.user_inputs = [[gender, yearlevel, course, income]]
#run the app
if __name__ == "__main__":
    app()