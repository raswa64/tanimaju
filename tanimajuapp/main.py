import streamlit as st
import numpy as np
from dataset_gathering import *
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("WoT Apps Pertanian Maju")
st.write(""" 
###### Smart Monitoring Tata Kelola Distribusi Pupuk di Kabupaten Indramayu
""")

dataset_name = st.sidebar.selectbox(
    'Pilih Dataset',
    ('Iris','Breast Cancer','Wine','XoR','Donut')
    )
st.write(f'### {dataset_name} Dataset')

classifier_name = st.sidebar.selectbox(
    'Pilih Kalsifikasi',
    ('k-NN','Perceptron','SVM','Random Forest')
)

def get_data(dataset_name):
    data = None
    if dataset_name=='Iris':
        data = datasets.load_iris()
        X,y = data.data, data.target
    elif dataset_name == 'Breast Canser':
        data = datasets.load_breast_cancer()
        X,y = data.data, data.target
    elif dataset_name == 'Wine':
        data = datasets.load_wine()
        X,y = data.data, data.target
    elif dataset_name == 'XoR':
        X,y = get_xor()
    else:
        X,y = get_donut()
    return X,y

X, y = get_data(dataset_name)

st.write('Ukuran Data: ', X.shape)
st.write('Jumlah label: ', len(np.unique(y)))

def parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
        params['kernel'] = st.sidebar.selectbox(
                         'Pilih Kernel',
                         ('linear', 'precomputed', 'poly', 'sigmoid', 'rbf'))
    elif clf_name == 'k-NN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name == 'Random Forest':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    else:
        params['penalty'] = st.sidebar.selectbox(
                            'Pilih Penalty',
                            ('l1','l2','elasticnet', None))
    return params

params = parameter_ui(classifier_name)

def get_classifier(clf_name, params):
     clf=None
     if clf_name == 'SVM':
        clf =SVC(C = params['C'], kernel = params['kernel'])
     elif clf_name == 'k-NN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
     elif clf_name == 'Random Forest':
        clf = RandomForestClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'])
     else:
        clf = Perceptron(penalty=params['penalty'])
     return clf

clf = get_classifier(classifier_name, params)

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state = 46) 
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test,y_pred)

st.write(f'Classifier:  {classifier_name}')
st.write(f'Akurasi: ', acc)

#PCA
def reduksi_dimensi(X):
    if dataset_name not in ['XoR', 'Donut']:
        pca = PCA(2)
        X_red = pca.fit_transform(X)
        return pca, X_red
    return None, X

pca, X_red = reduksi_dimensi(X)
fig = plt.figure()
plt.scatter(X_red[:,0],X_red[:,1],\
    c=y, alpha=0.8, cmap='RdYlBu')

plt.xlabel("Sumbu-X")
plt.ylabel("Sumbu-Y")

plt.colorbar()
st.pyplot(fig)

st.write('### Perbandingan y_pred & y_test')
if dataset_name not in ['XoR', 'Donut']:
    X_test_red = pca.transform(X_test)
else:
    X_test_red = X_test

st.write("#### y_test")
fig = plt.figure()
plt.scatter(X_test_red[:,0], X_test_red[:,1],\
    c=y_test, alpha=0.8, cmap='viridis')
    
plt.xlabel("Sumbu-X")
plt.ylabel("Sumbu-Y")

plt.colorbar()
st.pyplot(fig)    

st.write("#### y_pred")
fig = plt.figure()
plt.scatter(X_test_red[:,0],X_test_red[:,1],\
    c=y_pred, alpha=0.8, cmap='viridis')
    
plt.xlabel("Sumbu-X")
plt.ylabel("Sumbu-Y")

plt.colorbar()
st.pyplot(fig)    