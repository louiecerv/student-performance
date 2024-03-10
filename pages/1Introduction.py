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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

# Define the Streamlit app
def app():

    if "le_list" not in st.session_state:
        st.session_state.le_list = []

    st.subheader('The task: Classify respondent g-banking usage as either low, moderate or high.')
    text = """Describe the dataset and the various algorithms here."""
    st.write(text)

    df = st.session_state['df']
        
    # Create a progress bar object
    st.progress_bar = st.progress(0, text="Generating data graphs please wait...")

    st.write('Browse the dataset')
    st.write(df)

    le = LabelEncoder()

    #Get the list of column names
    column_names = features.columns.tolist()

    le_list = []  # Create an empty array to store LabelEncoders
    # Loop through each column name
    for cn in column_names:
        le = LabelEncoder()  # Create a new LabelEncoder for each column
        le.fit(features[cn])  # Fit the encoder to the specific column
        le_list.append(le)  # Append the encoder to the list
        features[cn] = le.transform(features[cn])  # Transform the column using the fitted encoder

    # save the label encoder to the session state
    st.session_state["le"] = le_list
    
    st.write('After encoding to numbers')
    st.write(df)
    
    X = df.drop('usagelevel', axis=1)
    y = df['usagelevel']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #save the values to the session state    
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test    

    for i in range(100):
        # Update progress bar value
        st.progress_bar.progress(i + 1)
        # Simulate some time-consuming task (e.g., sleep)
        time.sleep(0.01)
    st.success("Data visualization completed!")

#run the app
if __name__ == "__main__":
    app()
