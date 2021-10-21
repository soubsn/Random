
from tkinter import Tk 
from tkinter.filedialog import askopenfilename
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
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 

def flatten(t):
    return [item for sublist in t for item in sublist]

def import_data():
    #GUI to select specific files to be imported
    Tk().withdraw() 
    filename = askopenfilename() 
    return(filename)

def evaluation(y, y_hat,title):
    cm = confusion_matrix(y, y_hat)
    precision = precision_score(y, y_hat)
    recall = recall_score(y, y_hat)
    accuracy = accuracy_score(y,y_hat)
    f1 = f1_score(y,y_hat)
    res=[float(recall),float(accuracy),float(precision),float(f1)]
    return res
    
def cross_validate(classifier, cv, output):
    pipeline = Pipeline(steps=[('preprocess', preprocess),('classifier', classifier)])
    train_acc = []
    test_acc = []
    res=[]
    y_v =[]
    y_p = []
    for train_ind, val_ind in cv.split(X_train, y_train):
        X_t, y_t = X_train.iloc[train_ind], y_train.iloc[train_ind]
        pipeline.fit(X_t, y_t)
        y_hat_t = pipeline.predict(X_t)
        train_acc.append(accuracy_score(y_t, y_hat_t))
        X_val, y_val = X_train.iloc[val_ind], y_train.iloc[val_ind]
        y_hat_val = pipeline.predict(X_val)
        test_acc.append(accuracy_score(y_val, y_hat_val))
        r1 = evaluation(y_val, y_hat_val,classifier)
        res.append(r1)
        y_v.append(y_val['Classification'].tolist())
        y_p.append(y_hat_val.tolist())
        gc.collect
    result_statistics = pd.DataFrame(res, columns=['Recall', 'Accuracy', 'Precision','F1'])
    y_value=flatten(y_v)
    y_predict=flatten(y_p)
    cm = confusion_matrix(y_value, y_predict)
    precision = precision_score(y_value, y_predict)
    recall = recall_score(y_value, y_predict)
    accuracy = accuracy_score(y_value,y_predict)
    f1 = f1_score(y_value,y_predict)
    print(classifier)
    print('Recall: ', recall)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('F1: ', f1)
    sns.heatmap(cm,  cmap= 'PuBu', annot=True, fmt='g', annot_kws=    {'size':20})
    plt.xlabel('predicted', fontsize=18)
    plt.ylabel('actual', fontsize=18)
    plt.title(classifier, fontsize=18)
    plt.show();
    print('\n')
    res=[float(recall),float(accuracy),float(precision),float(f1)]
    if output == 0:
        return res
    else:
        return pipeline

def grid_search(classifier, param_grid, cv):
    search = GridSearchCV(Pipeline(steps=[
        ('preprocess', preprocess),
        ('classifier', classifier)
    ]), param_grid, cv=cv)
    train_acc = []
    test_acc = []
    res=[]
    y_v =[]
    y_p = []
    for train_ind, val_ind in cv.split(X_train, y_train):
        X_t, y_t = X_train.iloc[train_ind], y_train.iloc[train_ind]
        search.fit(X_t, y_t)
        y_hat_t = search.predict(X_t)
        train_acc.append(accuracy_score(y_t, y_hat_t))
        X_val, y_val = X_train.iloc[val_ind], y_train.iloc[val_ind]
        y_hat_val = search.predict(X_val)
        test_acc.append(accuracy_score(y_val, y_hat_val))
        r1 = evaluation(y_val, y_hat_val,classifier)
        res.append(r1)
        y_v.append(y_val['Classification'].tolist())
        y_p.append(y_hat_val.tolist())
        gc.collect
    y_value=flatten(y_v)
    y_predict=flatten(y_p)
    cm = confusion_matrix(y_value, y_predict)
    precision = precision_score(y_value, y_predict)
    recall = recall_score(y_value, y_predict)
    accuracy = accuracy_score(y_value,y_predict)
    f1 = f1_score(y_value,y_predict)
    print(classifier)
    print('Recall: ', recall)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('F1: ', f1)
    sns.heatmap(cm,  cmap= 'PuBu', annot=True, fmt='g', annot_kws=    {'size':20})
    plt.xlabel('predicted', fontsize=18)
    plt.ylabel('actual', fontsize=18)
    plt.title(classifier, fontsize=18)
    plt.show();
    print('\n')
    print('Training Accuracy: {}'.format(np.mean(train_acc)))
    print('\n')
    print('Validation Accuracy: {}'.format(np.mean(test_acc)))
    print('\n')
    print('Grid Search Best Params:')
    print('\n')
    print(search.best_params_)
    return search


####### The first pop up you must select the data label. The second popup is to select the data you want to train. 
####### The third is to select the testing data. The program will go though 4 different ml algorythm and select the best.
######  After the hyper-parameters will be tuned to best fit the training data. The final classifier is then trained, tested and saved.
 
# retrive data   
data_labels=pd.read_csv(import_data())
data_train=pd.read_csv(import_data())
data_test=import_data()
data_name=data_test
data_test=pd.read_csv(data_test)
#####

#filter data
data_train = data_train[data_train.columns.drop(list(data_train.filter(regex='Assay')))]
labels = list(data_train['Molecular Property'])
data_train = data_train[data_train.columns.drop(list(data_train.filter(regex='Molecular')))]
data_train = data_train.reindex(sorted(data_train.columns, key=int), axis=1)
data_train = data_train.T
data_train.columns=[labels]


data_test = data_test[data_test.columns.drop(list(data_test.filter(regex='Assay')))]
data_test = data_test[data_test.columns.drop(list(data_test.filter(regex='Molecular')))]
data_test = data_test.reindex(sorted(data_test.columns, key=int), axis=1)
data_test = data_test.T
data_test.columns=[labels]

data_labels=data_labels.replace(to_replace = 'Non Drug-Like', value =0)
data_labels=data_labels.replace(to_replace = 'Drug-Like', value =1)
data_labels = data_labels [data_labels .columns.drop(list(data_labels .filter(regex='Molecule')))]
data_labels_train = data_labels.loc[0:(min(data_train.shape)-1),:]
data_labels_test = data_labels.loc[min(data_train.shape):,:]
numerical_columns = data_train.keys()
#####

#preprocessing
#Creating ss transformer to scale the numerical data with StandardScaler()
ss = Pipeline(steps=[('ss', StandardScaler())])
preprocess = ColumnTransformer(transformers=[('cont', ss,numerical_columns)])  

#Testing the different models on data!!!
X_train, X_test, y_train, y_test = train_test_split(data_train, data_labels_train, random_state=42) 

t1=cross_validate(LinearSVC(class_weight = {0:0.6, 1:0.4}), RepeatedKFold(n_splits=6, n_repeats=20),0)
t2=cross_validate(GaussianNB(priors=[0.6, 0.4]), RepeatedKFold(n_splits=6, n_repeats=20),0)
t3=cross_validate(DecisionTreeClassifier(class_weight = {0:0.6, 1:0.4}) , RepeatedKFold(n_splits=6, n_repeats=20),0)
t4=cross_validate(KNeighborsClassifier() , RepeatedKFold(n_splits=6, n_repeats=20),0)

recall_sorter = pd.DataFrame({'Classifier':[1,2,3,4],'Result':[t1[0],t2[0],t3[0],t4[0]]})
f1_sorter = pd.DataFrame({'Classifier':[1,2,3,4],'Result':[t1[3],t2[3],t3[3],t4[3]]})

max_recall_sorter_value = recall_sorter['Result'].idxmax()
max_f1_sorter_value = f1_sorter['Result'].idxmax()
gc.collect
#####

#Selecting best sorter based on F1 and recall. If there is no agreement, script will end for examination of data
if max_recall_sorter_value == max_f1_sorter_value:
    if max_recall_sorter_value == 0:
        params = {'classifier__C': [0.01, 0.1, 1],'classifier__max_iter': [10 ,50, 100]}
        classifier = LinearSVC(class_weight = {0:0.643, 1:0.357})
    elif max_recall_sorter_value == 1:
        params = {'classifier__var_smoothing': [0.000000000001, 0.0000000001,0.00000001]}
        classifier = GaussianNB(priors=[0.6, 0.4])
    elif max_recall_sorter_value == 2:
        params = {'classifier__min_samples_split': [2, 3, 4, 5],'classifier__min_samples_leaf': [1, 2, 5], 'classifier__max_features':[5,50,500,5000]}
        classifier = DecisionTreeClassifier(class_weight = {0:0.643, 1:0.357})
    elif max_recall_sorter_value == 3:
        params = {'classifier__leaf_size': [10, 20, 30, 40, 50],'classifier__algorithm': ['auto','ball_tree','kd_tree']}
        classifier = KNeighborsClassifier() 

#Calling the grid_search function to test the parameters above
p1=grid_search(classifier, params, RepeatedKFold(n_splits=6, n_repeats=3))
gc.collect
#####

#Training the real pipeline
if max_recall_sorter_value == max_f1_sorter_value:
    if max_recall_sorter_value == 0:
        final_classifier = LinearSVC(class_weight = {0:0.643, 1:0.357}, C=p1.best_params_['classifier__C'],max_iter= p1.best_params_['classifier__max_iter'])
    elif max_recall_sorter_value == 1:
        final_classifier = GaussianNB(priors=[0.6, 0.4],var_smoothing= p1.best_params_['classifier__var_smoothing'])
    elif max_recall_sorter_value == 2:
        final_classifier = DecisionTreeClassifier(class_weight = {0:0.643, 1:0.357}, min_samples_split= p1.best_params_['classifier__min_samples_split'], min_samples_leaf=p1.best_params_['classifier__min_samples_leaf'], max_features=p1.best_params_['classifier__max_features'])
    elif max_recall_sorter_value == 3:
        final_classifier =KNeighborsClassifier(leaf_size=p1.best_params_['classifier__leaf_size'], algorithm=p1.best_params_['classifier__algorithm']) 

final_model=cross_validate(final_classifier,RepeatedKFold(n_splits=6, n_repeats=20),1)
gc.collect
#####

#Test final pipeline
data_predict= final_model.predict(data_test)
cm = confusion_matrix(data_labels_test, data_predict)
precision = precision_score(data_labels_test, data_predict)
recall = recall_score(data_labels_test, data_predict)
accuracy = accuracy_score(data_labels_test,data_predict)
f1 = f1_score(data_labels_test,data_predict)
print('Recall: ', recall)
print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('F1: ', f1)
sns.heatmap(cm,  cmap= 'PuBu', annot=True, fmt='g', annot_kws=    {'size':20})
plt.xlabel('predicted', fontsize=18)
plt.ylabel('actual', fontsize=18)
plt.title(final_classifier, fontsize=18)
plt.show();
#####

#Save Pipeline for further use
data_name=data_name.split("/")
data_name[-1]='finalized_model_Nic_LTX_ML_Challenge.sav'
data_name="/".join(data_name)
pickle.dump(final_model, open(data_name, 'wb'))
