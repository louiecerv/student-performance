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

    st.subheader('The task: Classify respondent g-banking usage as either low or high.')
    text = """Describe the dataset and the various algorithms here."""
    st.write(text)

    df = st.session_state['df']

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
        
    # Create a progress bar object
    progress_bar = st.progress(0, text="Generating data graphs please wait...")
    
    fig, ax = plt.subplots()
    # Create the horizontal barplot
    sns.countplot(y='Sex', data=df, hue='Sex', palette='bright', ax=ax)
    # Add the title
    ax.set_title('Plot of Sex Distribution')
    # Display the plot using Streamlit
    st.pyplot(fig)

    # Create a new figure and axes object
    fig, ax = plt.subplots()
    # Create a horizontal barplot using seaborn
    sns.countplot(y='Income', data=df, hue='Income', palette='bright', ax=ax)
    # Set title
    ax.set_title('Plot of Income Distribution')
    st.pyplot(fig)

    # Create a new figure and axes object
    fig, ax = plt.subplots()
    # Create a horizontal barplot using seaborn
    sns.countplot(y='usagelevel', data=df, hue='usagelevel', palette='bright', ax=ax)

    # Set title
    ax.set_title('Plot of E-banking Usage Distribution')
    st.pyplot(fig)

    for i in range(100):
        # Update progress bar value
        progress_bar.progress(i + 1)
        # Simulate some time-consuming task (e.g., sleep)
        time.sleep(0.01)
    st.success("Data visualization completed!")

#run the app
if __name__ == "__main__":
    app()
