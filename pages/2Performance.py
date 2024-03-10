#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


# Define the Streamlit app
def app():

    classifier = ''
    if st.session_state['selected_model'] == 0:     # Logistic Regression
        report = """Achieves good accuracy, but can be prone to 
        overfitting, leading to lower performance on unseen data.
        Simple and interpretable, allowing visualization of decision rules.
        Susceptible to changes in the training data, potentially 
        leading to high variance in predictions."""
        classifier = 'Logistic Regression'
    elif st.session_state['selected_model'] == 1:   # SVR
        report = """Performance: Can achieve similar or slightly better 
        accuracy compared to a random forest, but results can vary 
        depending on hyperparameter tuning. Introduces additional randomness 
        during tree building by randomly selecting features at each split.  Aims to 
        further improve generalization and reduce overfitting by increasing 
        the diversity of trees in the ensemble. Requires careful 
        hyperparameter tuning to achieve optimal performance."""
        classifier = "SVR Regressor"
    elif st.session_state['selected_model'] == 2:   # Decision Tree
        report = """Performance: Can achieve similar or slightly better 
        accuracy compared to a random forest, but results can vary 
        depending on hyperparameter tuning. Introduces additional randomness 
        during tree building by randomly selecting features at each split.  Aims to 
        further improve generalization and reduce overfitting by increasing 
        the diversity of trees in the ensemble. Requires careful 
        hyperparameter tuning to achieve optimal performance."""
        classifier = "Decision Tree"
    else:   #Gradient Boosting
        report = """Gradient boosting classifiers perform well on the MNIST 
        handwritten digit dataset, typically achieving accuracy in the high 90s 
        (often exceeding 98%). Gradient boosting excels at handling complex, 
        non-linear relationships within the data, which is helpful for
        recognizing the varied shapes of handwritten digits. It can achieve 
        accuracy comparable to other popular methods for MNIST like 
        Support Vector Machines (SVMs) and Random Forests."""
        classifier = "Gradient Boosting"

    st.subheader('Performance of the ' + classifier)

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train= st.session_state['y_train']
    y_test = st.session_state['y_test']

    clf = LogisticRegression()
    clf = SVR()
    DecisionTreeRegressor()
    GradientBoostingRegressor()

    st.sidebar.subheader('Select the classifier')

    # Create the selection of classifier
    clf = LogisticRegression()
    st.session_state['selected_model'] = 0
    options = ['Logistic Regression', 'SVR Regressor', 'Decision Tree', 'Gradient Boosting']
    selected_option = st.sidebar.selectbox('Select the classifier', options)
    if selected_option=='SVR Regressor':        
        clf = SVR()
        st.session_state['selected_model'] = 1
    elif selected_option == 'Decision Tree':
        clf = DecisionTreeRegressor()
        st.session_state['selected_model'] = 2
    elif selected_option == 'Gradient Boosting':
        clf = GradientBoostingRegressor()
        st.session_state['selected_model'] = 3

    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)

    st.subheader('R-squared')
    r2 = r2_score(y_test, y_test_pred)
    st.write(f"R-squared: {r2:.4f}")

    st.subheader('Mean Squared Error (MSE)')
    mse = mean_squared_error(y_test, y_test_pred)
    st.write(f"MSE: {mse:.4f}")
    
    st.write(classifier)
    st.write(report)

    # save the clf to the session state
    st.session_state['clf'] = clf

#run the app
if __name__ == "__main__":
    app()
