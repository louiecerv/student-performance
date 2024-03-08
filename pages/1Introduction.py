#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml

# Define the Streamlit app
def app():
    st.session_state["current_form"] = 2

    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

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

    st.text('The task: Classify handwritten digits from 0 to 9 based on a given image.')
    text = """Dataset: MNIST - 70,000 images of handwritten digits (28x28 pixels), each labeled 
    with its corresponding digit (0-9).
    \nModels:
    \nK-Nearest Neighbors (KNN):
    \nEach image is represented as a 784-dimensional vector (28x28 pixels). 
    To classify a new image, its distance is measured to K nearest neighbors in the 
    training data. The majority class label among the neighbors is assigned to the new image.
    \nDecision Tree:
    \nA tree-like structure is built based on features (pixel intensities) of the images. 
    \nThe tree splits the data based on decision rules (e.g., "pixel intensity at 
    position X is greater than Y"). The new image is navigated through the tree based on 
    its features, reaching a leaf node representing the predicted digit class.
    \nRandom Forest:
    \nAn ensemble of multiple decision trees are built, each trained on a random subset of 
    features (pixels) and a random subset of data.
    \nTo classify a new image, it is passed through each decision tree, and the majority class 
    label from all predictions is assigned."""
    st.write(text)

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']

    st.subheader('First 25 images in the MNIST dataset') 

    # Get the first 25 images and reshape them to 28x28 pixels
    train_images = np.array(X_train)
    train_labels = np.array(y_train)
    images = train_images[:25].reshape(-1, 28, 28)
    # Create a 5x5 grid of subplots
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    # Plot each image on a separate subplot
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(images[i], cmap=plt.cm.binary)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Digit: {train_labels[i]}")
    # Show the plot
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader('Select the classifier')

    # Create the selecton of classifier

    clf = tree.DecisionTreeClassifier()
    options = ['Decision Tree', 'Random Forest Classifier', 'Extreme Random Forest Classifier', 'K Nearest Neighbor']
    selected_option = st.selectbox('Select the classifier', options)
    if selected_option =='Random Forest Classifier':
        clf = RandomForestClassifier(n_jobs=2, random_state=0)
        st.session_state['selected_model'] = 1
    elif selected_option=='Extreme Random Forest Classifier':        
        clf = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=0)
        st.session_state['selected_model'] = 2
    elif selected_option == 'K Nearest Neighbor':
        clf = KNeighborsClassifier(n_neighbors=5)
        st.session_state['selected_model'] = 3
    else:
        clf = tree.DecisionTreeClassifier()
        st.session_state['selected_model'] = 0

    # save the clf to the session variable
    st.session_state['clf'] = clf

    if st.button("Training"):
        st.switch_page("/pages/2Training") 

#run the app
if __name__ == "__main__":
    app()
