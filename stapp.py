import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
# Set title
st.title("Welcome to Machine Learning App")
image=Image.open('mlapp.jpg')
st.image(image,use_column_width=True)
#set subtitle
st.markdown('<h2 style="color:red;">"A Data App which automates the working of Machine Learning process"</h2>', unsafe_allow_html=True)
st.markdown('<h2 style="color:grey;">    "Lets Explore different datasets and classifiers"</h2>', unsafe_allow_html=True)
dataset_name=st.sidebar.selectbox("Select Dataset",("Breast Cancer","Iris","Wine"))
classifier_name=st.sidebar.selectbox("Select Classifier",("KNN","SVM"))
def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    Y = data.target
    return X, Y

X, Y = get_dataset(dataset_name)
st.dataframe(X)
st.write("Shape of your data:", X.shape)
st.write("Unique Target Variables:", len(np.unique(Y)))
st.set_option('deprecation.showPyplotGlobalUse', False)
fig, ax1 = plt.subplots()
sns.boxplot(data=X, ax=ax1)
ax1.set_title('Figure 1')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Count')
st.pyplot()

# Plotting on the second figure (ax2)
fig, ax2 = plt.subplots()
sns.histplot(X, ax=ax2)
ax2.set_title('Figure 1')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Count')
st.pyplot()
def add_parameter(name_of_clf):
    params=dict()
    if name_of_clf=="SVM":
        c=st.sidebar.slider('C',.01,15.0)
        params['C']=c
    elif name_of_clf=="KNN":
        k=st.sidebar.slider('K',1,10)
        params["K"]=k    
    return params
params=add_parameter(classifier_name)
def get_classifier(name_of_clf,params):
    clf=None
    if name_of_clf=="SVM":
        clf=SVC(C=params['C'])
    else:
        clf=KNeighborsClassifier(n_neighbors=params['K'])
    return clf
clf=get_classifier(classifier_name,params)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
accuracy=accuracy_score(Y_test,y_pred)
st.write("Predictions",y_pred)
st.write("classifier name:",classifier_name)
st.write("Accuracy of your model is",accuracy)    

