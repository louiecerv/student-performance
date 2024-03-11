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

"""
    st.write("""The trained model will predict the Usage Level from the information provided.""")
    st.subheader('User Information')
    gender = form3.selectbox('Gender:', ['Male', 'Female'])
    st.session_state['gender'] = gender
    age = form3.selectbox('Age:', ['21-25', '16-20', '11-15', '26-30', '6-10', '1-5'])
    st.session_state['age'] = age
    educlevel = form3.selectbox('Education Level:', ['School', 'College', 'University'])

    """
    
#run the app
if __name__ == "__main__":
    app()