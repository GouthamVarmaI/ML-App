import streamlit as st
from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("The App that does Machine Learning!")

st.write("""Use the slider at the left to give your inputs...""")

dataset_name = st.sidebar.selectbox(" Select the Data",("Iris", "Breast Cancer","Wine Dataset"))

classifier_name = st.sidebar.selectbox(" Select the Classifier",("KNN", "SVM","Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return(x, y)

x, y = get_dataset(dataset_name)
st.write("Size of the data", x.shape)
st.write("Number of classes", len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K value", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.1, 10.0)
        params["C"] = C
    else :
        max_depth = st.sidebar.slider("Max Depth",2,15)
        n_estimators = st.sidebar.slider("N Estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return(params)

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else :
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"], random_state = 1)
    return(clf)

clf = get_classifier(classifier_name, params)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)

clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)

acc = accuracy_score(y_test, y_predict)
st.write(f"Classifier used =  {classifier_name}")
st.write(f"Accuracy =  {acc}")

if acc>0.85:
    st.write("Yay! Great accuracy")
elif acc>0.7:
    st.write("Could get better at it!")
else:
    st.write("Nah! Try different options and see if it helps...")
