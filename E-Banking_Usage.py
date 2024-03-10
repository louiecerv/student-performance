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
from sklearn.datasets import fetch_openml
import time

# Define the Streamlit app
def app():


    text = """E-Banking Usage Among CBM Students"""
    st.subheader(text)

    # Use session state to track the current form
    if "clf" not in st.session_state: 
        st.session_state["clf"] = []

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
    
    if "mnist" not in st.session_state: 
        st.session_state["mnist"] = []

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('ebabaking.jpg', caption="The E-banking Usage")
    text = """Dataset for E-Banking Usage and Spending Habits of CBM Students at WVSU
    \nThis dataset investigates the relationship between e-banking usage and 
    spending habits among College of Business Management (CBM) students at 
    West Visayas State University (WVSU).
    \nFeatures:
    \nfamily_income (categorical): This feature represents the student's family 
    income level. It is divided into categories like "low," "medium," or "high" based
    on a pre-defined income range.
    \nSex (binary): This feature indicates the student's sex, coded as "male" or "female."
    \ncourse (categorical): This feature specifies the student's academic program within CBM. 
    Label: e_banking_usage (binary): This variable indicates the student's level of 
    e-banking usage. It is coded as "low" or "high"."""

    st.write(text)

    if "dataset_ready" not in st.session_state:
        # Create a progress bar object
        progress_bar = st.progress(0, text="Loading the dataset please wait...")

        #replace with your dataset
        df = pd.read_csv('encoded-ebanking.csv', header=0)

        # load the data and the labels
        X = df.values[:,0:-1]
        y = df.values[:,-1]        

        st.subheader('The Dataset')
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Dataset loading completed!")

        st.session_state.dataset_ready = True

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #save the values to the session state    
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test    

    
#run the app
if __name__ == "__main__":
    app()
