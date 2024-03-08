#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():

    if "reset_app" not in st.session_state:
        st.session_state.reset_app = False

    text = """Decision Tree, Random Forest and K-Nearest Neighbor on the MNIST Dataset"""
    st.subheader(text)

    # Use session state to track the current form
    if "current_form" not in st.session_state:
        st.session_state["current_form"] = 0

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

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('MNIST.png', caption="Modified National Institute of Standards and Technology", use_column_width=True)
    text = """MNIST is a large database of handwritten digits that is commonly used for training and
    testing various image processing systems. The acronym stands for Modified National Institute 
    of Standards and Technology. MNIST is a popular dataset in the field of machine learning and 
    can provide a baseline for benchmarking algorithms."""
    st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
