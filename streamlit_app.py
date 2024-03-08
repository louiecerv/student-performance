#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
import time

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

    if "dataset_ready" not in st.session_state:
        # Create a progress bar object
        progress_bar = st.progress(0, text="Loading the dataset please wait...")
        
        # Load MNIST dataset
        mnist = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

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
    X, y = mnist
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
