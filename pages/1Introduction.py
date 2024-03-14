#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
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

    st.subheader('The task: Classify E-banking usage as very low, low, moderate, high or very high.')
    text = """CBM Student E-Banking Usage Dataset
    \nThis dataset investigates the factors that affect e-banking usage and spending habits 
    among students at CBM.
    Features:
    sex (categorical): Student's gender (e.g., Male, Female)
    year_level (categorical): Student's year in the program
    course (categorical): Student's course of study
    family_income (numerical): Student's reported family income level
    Target Variable:
    usagelevel (categorical): Level of e-banking usage by the student
    Sampling Method:
    Stratified random sampling: This ensures the sample population reflects the 
    proportions of students based on year level and/or course within CBM."""
    with st.expander("About the Dataset. CLick to expand."):
        st.write(text)

    #replace with your dataset
    df = pd.read_csv('hei-perf.csv', header=0)
    
    # Shuffle the DataFrame (returns a copy)
    df = df.sample(frac=1)
    st.write('Browse the dataset')
    st.write(df)

    # Get column names and unique values
    columns = df.columns
    unique_values = {col: df[col].unique() for col in columns}    
    
    # Display unique values for each column
    st.write("\n**Unique Values:**")
    for col, values in unique_values.items():
        st.write(f"- {col}: {', '.join(map(str, values))}")

    st.write('Descriptive Statistics')
    st.write(df.describe().T)
    df = df.drop('STUDENT ID', axis=1)
    X = df.drop('GRADE', axis=1)
    y = df['GRADE']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    #save the values to the session state    
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test    
    

#run the app
if __name__ == "__main__":
    app()
