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


    text = """E-banking Usage level and Influence on Spending Habits Among College of Business and Management Students"""
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

    st.image('e-banking.jpg', caption="The E-banking Usage")
    text = """Dataset for E-Banking Usage and Spending Habits of CBM Students at WVSU
    \nThis dataset investigates the factors that affect the e-banking usage and 
    spending habits among College of Business Management (CBM) students at 
    West Visayas State University (WVSU).
    \nFeatures:
    \nfamily_income (categorical): This feature represents the student's family 
    income level. It is divided into categories based on a pre-defined income range.
    \nSex (binary): This feature indicates the student's sex, coded as "male" or "female."
    \ncourse (categorical): This feature specifies the student's academic program within CBM. 
    \nLabel: e_banking_usage (binary): This variable indicates the student's level of 
    e-banking usage. It is coded as categories of 'very high', 'high', 'moderate', 'low' and 'very low'."""

    st.write(text)

#run the app
if __name__ == "__main__":
    app()
