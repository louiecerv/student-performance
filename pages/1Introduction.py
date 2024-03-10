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

    st.subheader('The task: Classify respondent g-banking usage as either low or high.')
    text = """Describe the dataset and the various algorithms here."""
    st.write(text)

    df = st.session_state['df']

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
        
    # Create a progress bar object
    st.progress_bar = st.progress(0, text="Generating data graphs please wait...")

    st.write('Browse the dataset')
    st.write(df)

    le = LabelEncoder()

      # Separate features and target column
    features = df.drop(target_column, axis=1)
    target = df[target_column]

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
    # Combine encoded features and target column
    df = pd.concat([features, target], axis=1)
    
    st.write('After encoding to numbers')
    st.write(df)

    for i in range(100):
        # Update progress bar value
        st.progress_bar.progress(i + 1)
        # Simulate some time-consuming task (e.g., sleep)
        time.sleep(0.01)
    st.success("Data visualization completed!")

#run the app
if __name__ == "__main__":
    app()
