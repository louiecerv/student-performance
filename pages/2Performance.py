#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Define the Streamlit app
def app():

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train= st.session_state['y_train']
    y_test = st.session_state['y_test']
    clf = st.session_state['clf']

    st.sidebar.subheader('Select the classifier')

    # Create the selection of classifier
    options = ['Extra Trees Classifier', 'SVM', 'Decision Tree', 'Gradient Boosting']
    selected_option = st.sidebar.selectbox('Select the classifier', options)
    if selected_option=='Extra Trees Classifier':        
        clf = ExtraTreesClassifier()
        st.session_state['selected_model'] = 0
    elif selected_option=='SVM':        
        clf = SVC()
        st.session_state['selected_model'] = 1
    elif selected_option == 'Decision Tree':
        clf = DecisionTreeClassifier()
        st.session_state['selected_model'] = 2
    elif selected_option == 'Gradient Boosting':
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3)
        st.session_state['selected_model'] = 3

    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)

    st.subheader('Confusion Matrix')
    st.write('Confusion Matrix')
    cm = confusion_matrix(y_test, y_test_pred)
    st.text(cm)

    st.subheader('Performance Metrics')
    st.text(classification_report(y_test, y_test_pred))

    classifier = ''
    if st.session_state['selected_model'] == 0:     
        classifier = 'Extra Trees Classifier'
        report = """Extra Trees Classifier is a machine learning method for classification tasks. 
        It belongs to a family of algorithms called ensemble methods, which combine multiple
         models (in this case, decision trees) to get a better prediction than any single model.
         It builds multiple decision trees, but with randomness injected at two key points: 
         selecting features and choosing split points within those features. This randomness 
         helps reduce overfitting by making the trees more independent."""
        
    elif st.session_state['selected_model'] == 1:   
        classifier = "Support Vector Machine"
        report = """A support vector machine (SVM) is a machine learning algorithm that excels at 
        classifying data points.  SVMs find te line (or hyperplane in higher dimensions) 
        that maximizes the margin between the two classes. A wider margin translates to a 
        better separation and potentially better classification on unseen data. SVMs are 
        particularly useful for: High dimensional data: They can handle complex data 
        with many features. Smaller datasets: They can perform well even with limited 
        training data. Non-linear classification: By using kernel functions, 
        SVMs can handle data that isn't easily separable with a straight line."""
        
    elif st.session_state['selected_model'] == 2:  
        classifier = "Decision Tree"
        report = """A decision tree classifier is a machine learning algorithm that works 
        by building a tree-like model to classify data. It asks a series of questions 
        about the data's features at each branch of the tree, and based on the answers, 
        it directs the data down a specific path until it reaches a leaf node that 
        contains the predicted classification. 
        Easy to understand and interpret: The decision-making process is clear and 
        transparent, unlike some other machine learning models. Good for handling 
        high-dimensional data: They can work well with data that has many features. 
        Fast training time: Compared to some other algorithms, decision trees can be 
        trained relatively quickly."""

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

    st.write(classifier)
    st.write(report)

    # save the clf to the session state
    st.session_state['clf'] = clf

#run the app
if __name__ == "__main__":
    app()
