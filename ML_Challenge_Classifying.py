#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 14:35:31 2021

@author: nicolassoubry
"""

from tkinter import Tk 
from tkinter.filedialog import askopenfilename, messagebox
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')
import gc

# Transformers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

# Modeling Evaluation
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, classification_report, RocCurveDisplay
from IPython.display import display, Markdown

# Pipelines
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

# Machine Learning
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 

def import_data():
    #GUI to select specific files to be imported
    Tk().withdraw() 
    filename = askopenfilename() 
    return(filename)

####### The first pop up you must select the data you want to test. The second popup is to select the model you have trained. 
####### The third is to select labels for the data if you have it. If not an error will be generated but will will have the 
####### outcome of the classifying printed and saved.

#Retrieve and filter data
data_test=import_data()
data_name=data_test
data_test=pd.read_csv(data_test)
loaded_model = pickle.load(open(import_data(), 'rb'))
data_test = data_test[data_test.columns.drop(list(data_test.filter(regex='Assay')))]
labels = list(data_test['Molecular Property'])
data_test = data_test[data_test.columns.drop(list(data_test.filter(regex='Molecular')))]
data_test = data_test.reindex(sorted(data_test.columns, key=int), axis=1)
data_test = data_test.T
data_test.columns=[labels]


data_predict= loaded_model.predict(data_test)
data_predict_labelled = pd.DataFrame(data_predict, columns=['Classification'])
data_predict_labelled=data_predict_labelled.replace(to_replace = 0, value='Non Drug-Like')
data_predict_labelled=data_predict_labelled.replace(to_replace = 1, value ='Drug-Like')
print(data_predict_labelled) 

data_name=data_name.split("/")
data_name[-1]='Data_Classified_Nic_LTX_ML_Challenge.csv'
data_name="/".join(data_name)
data_predict_labelled.to_csv(data_name, index=True)

#Check how models compares to known labels. Will throw error if unknown
data_labels=pd.read_csv(import_data())
data_labels=data_labels.replace(to_replace = 'Non Drug-Like', value =0)
data_labels=data_labels.replace(to_replace = 'Drug-Like', value =1)
data_labels = data_labels [data_labels .columns.drop(list(data_labels .filter(regex='Molecule')))]

cm = confusion_matrix(data_labels , data_predict)
precision = precision_score(data_labels , data_predict)
recall = recall_score(data_labels , data_predict)
accuracy = accuracy_score(data_labels ,data_predict)
f1 = f1_score(data_labels ,data_predict)
print('Recall: ', recall)
print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('F1: ', f1)
print()
sns.heatmap(cm,  cmap= 'PuBu', annot=True, fmt='g', annot_kws=    {'size':20})
plt.xlabel('predicted', fontsize=18)
plt.ylabel('actual', fontsize=18)
plt.title('loaded_model', fontsize=18)
plt.show();

data_name=data_name.split("/")
data_name[-1]='finalized_model_Nic_LTX_ML_Challenge.sav'
data_name="/".join(data_name)
