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
from tensorflow.keras.datasets import mnist
import time

# Define the Streamlit app
def app():


    text = """Gradient Boosting Classifier on the MNIST Digits Dataset"""
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

    st.image('MNIST.png', caption="The MNIST Digits Dataset")

    if "dataset_ready" not in st.session_state:
        # Create a progress bar object
        progress_bar = st.progress(0, text="Loading the dataset please wait...")

        #replace with your dataset
        #Load MNIST dataset
        #st.session_state.mnist = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)
        st.session_state.mnist = mnist.load_data()
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)

        # Progress bar reaches 100% after the loop completes
        st.success("Dataset loading completed!")
        st.session_state.dataset_ready = True

        # Extract only the specified number of images and labels
        size = 10000
        X, y = st.session_state.mnist
        X = X[:size]
        y = y[:size]

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
