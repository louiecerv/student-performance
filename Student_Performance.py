#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import time

# Define the Streamlit app
def app():


    text = """Student Performance Classification Using Artificial Intelligence Techniques"""
    st.subheader(text)

    # Use session state to track the current form
    if "clf" not in st.session_state: 
        st.session_state["clf"] = []

    if "df" not in st.session_state: 
        st.session_state["df"] = []

    if "X_train" not in st.session_state: 
        st.session_state["X_train"] = []

    if "X_test" not in st.session_state: 
        st.session_state["X_test"] = []
    
    if "y_train" not in st.session_state: 
        st.session_state["X_train"] = []
    
    if "y_test" not in st.session_state: 
        st.session_state["y_yest"] = []

    if "selected_model" not in st.session_state: 
        st.session_state["selected_model"] = 0
    
    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('academic-perf.png', caption="Improving Student Performance Using AI")
    text = """This Data App uses the publicly available dataset found at https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation"""
    st.write(text)  
   
    text = """The original paper used artificial intelligence techniques are applied to the 
    questionnaire results that consists the main indicators, of three different courses of 
    two faculties in order to classify students’ final grade performances and to determine 
    the most efficient machine learning algorithm for this task."""
    with st.expander("Click to view Data App Description"):
        st.write(text)


#run the app
if __name__ == "__main__":
    app()
