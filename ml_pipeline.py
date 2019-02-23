#load packages

import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import seaborn as sns
import itertools

# classifiers / models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

# data
from sklearn import datasets

# other
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
import nltk
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import confusion_matrix, roc_curve, f1_score, roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
# data load
def load_data(path):
    '''
    load data (csv) from using a file path

    Parameters
    ----------
    path : str
        file path

    Returns
    -------
    DataFrame

        loaded dataset

    '''

    data_df=pd.read_csv(path)
    data_df=data_df._get_numeric_data()

    return data_df

def outlier_removal(data_df):
    '''
    remove outliers from the data

    Parameters
    ----------
    data_df : DataFrame

        data from which the outliers need to be removed

    Returns
    -------
    DataFrame

        data with outliers removed

    '''
    # removing values outside six standard deviations
    data_df=data_df[(np.abs(stats.zscore(data_df)) < 6).all(axis=1)]
    return data_df


def data_summary(data_df):
    '''
    descriptive statistics of data

    Parameters
    ----------
    data_df : DataFrame

        data to be analyzed

    Returns
    -------
    summary_df: DataFrame

        min, max, quantiles of all numerical columns

    corr_df: DataFrame

        pairs of columns with significant correlation

    '''
    # descriptive stats
    summary_df=data_df.describe()

    # correlation matrix
    data_df=data_df._get_numeric_data()
    rows, cols = data_df.shape
    flds = list(data_df.columns)
    corr = data_df.corr().values
    index=['column_x','column_y','corr']
    corr_df=pd.DataFrame(columns=index)
    for i in range(cols):
        for j in range(i+1, cols):
            if corr[i,j] > 0.8:
                temp=flds[i], flds[j], corr[i,j]
                corr_df=corr_df.append(pd.Series(temp,index=index),ignore_index=True)

    return summary_df,corr_df


def data_split(data_df,response='target',test_size=0.2,validate_size=0.2,random_state=None):
    '''
    split data into test and train

    Parameters
    ----------
    data_df : DataFrame

        data to be analyzed

    response: str

        name of the column to be predicted

    test_size: float

        proportion of test data

    validate_size: float

        proportion of validate data

    random_state: int

        set seed

    Returns
    -------
    Xtrain: DataFrame

        training data set

    Xvalidate: DataFrame

        validation data set

    Xtest: DataFrame

         test data set

    ytrain: DataFrame

        target for training data

    yvalidate: DataFrame

        target for validation data

    ytest: DataFrame

        target for test data

    '''

    data_df.rename(columns={response: 'target'}, inplace=True)

    feature_cols = [e for e in list(data_df) if e not in 'target']

    X= data_df.loc[:, feature_cols]
    y= data_df.target

    # splitting into test and train
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # splitting into train and validate
    Xtrain, Xvalidate, ytrain, yvalidate = train_test_split(Xtrain, ytrain, test_size=validate_size, random_state=random_state)
    return Xtrain, Xvalidate, Xtest, ytrain, yvalidate, ytest

def fit_classifers(Xtrain, Xvalidate, Xtest,ytrain, yvalidate, ytest,classifiers):
    '''
    split data into test and train

    Parameters
    ----------
    Xtrain: DataFrame

        training data set

    Xvalidate: DataFrame

        validation data set

    Xtest: DataFrame

         test data set

    ytrain: DataFrame

        target for training data

    yvalidate: DataFrame

        target for validation data

    ytest: DataFrame

        target for test data

    classifers: dict

        dictionary of classifiers

    Returns
    -------
    DataFrame

        containing scores to compare classifers

    '''
    train_scores = dict()
    valid_scores = dict()
    test_scores = dict()
    training_times = dict()

    for classifier_name, classifier_obj in classifiers.items():
        t = time.time()
        classifier_obj.fit(Xtrain, ytrain)

        training_times[classifier_name] = round(time.time() - t,2)
        train_scores[classifier_name] = round(classifier_obj.score(Xtrain, ytrain)*100,2)
        valid_scores[classifier_name] = round(classifier_obj.score(Xvalidate, yvalidate)*100,2)
        test_scores[classifier_name] = round(classifier_obj.score(Xtest, ytest)*100,2)


    df = pd.DataFrame([training_times,train_scores,valid_scores,test_scores]).T
    df.rename(columns={0:'Training Time',1:'Training Accuracy',2:'Validation Accuracy',3:'Test Accuracy'},inplace=True)
    return df


def feature_transformation(Xtrain, Xvalidate, Xtest):
    '''
    split data into test and train

    Parameters
    ----------
    Xtrain: DataFrame

        training data set

    Xvalidate: DataFrame

        validation data set

    Xtest: DataFrame

         test data set


    Returns
    -------
    Xtrain_new: DataFrame

        scaled training data set

    Xvalidate_new: DataFrame

        scaled validation data set

    Xtest_new: DataFrame

        scaled test data set

    '''

    # making a MinMaxScaler object
    scaler = MinMaxScaler()
    sc=scaler.fit(Xtrain)
    Xtrain_new=sc.transform(Xtrain)
    Xvalidate_new=sc.transform(Xvalidate)
    Xtest_new=sc.transform(Xtest)

    return Xtrain_new,Xvalidate_new,Xtest_new

def feature_selection(Xtrain, Xvalidate, Xtest, ytrain, yvalidate, ytest):
    '''
    split data into test and train

    Parameters
    ----------
    Xtrain: DataFrame

        training data set

    Xvalidate: DataFrame

        validation data set

    Xtest: DataFrame

         test data set

    ytrain: DataFrame

        target for training data

    yvalidate: DataFrame

        target for validation data

    ytest: DataFrame

        target for test data

    classifers: dict

        dictionary of classifiers

    Returns
    -------
    Xtrain_new: DataFrame

        selected features in training data set

    Xvalidate_new: DataFrame

        selected features in validation data set

    Xtest_new: DataFrame

        selected features in test data set

    '''

    rfe_scores=[]
    features=range(1,Xtrain.shape[1]+1)
    # looping over number of features
    for i in features:
        rfe = RFE(estimator = Ridge(), n_features_to_select = i)
        rfe.fit(Xtrain,ytrain)
        Xtrain_new=Xtrain[:,rfe.support_]
        Xvalidate_new=Xvalidate[:,rfe.support_]
        lr=LogisticRegression()
        lr.fit(Xtrain_new, ytrain)
        rfe_scores.append(round((1-lr.score(Xvalidate_new, yvalidate))*100,2))
    plt.plot(features,rfe_scores,label="score",linestyle='--', marker='o', color='b')
    plt.xlabel("Degree")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.show()
    print("The validation error is lowest when features=",features[np.argmin(rfe_scores)])

    rfe = RFE(estimator = Ridge(), n_features_to_select = features[np.argmin(rfe_scores)])
    rfe.fit(Xtrain,ytrain)
    Xtrain_new=Xtrain[:,rfe.support_]
    Xvalidate_new=Xvalidate[:,rfe.support_]
    Xtest_new=Xtest[:,rfe.support_]

    return Xtrain_new,Xvalidate_new,Xtest_new


def hyperparamter_optimization(Xtrain,Xvalidate,ytrain,yvalidate,classifiers,parameters):

    '''
    split data into test and train

    Parameters
    ----------
    Xtrain: DataFrame

        training data set

    Xvalidate: DataFrame

        validation data set

    Xtest: DataFrame

         test data set

    ytrain: DataFrame

        target for training data

    yvalidate: DataFrame

        target for validation data

    ytest: DataFrame

        target for test data

    classifers: dict

        dictionary of classifiers

    paramters: dict

        dictionary of paramters specific for classifiers

    Returns
    -------
    dict

        containing classifers with optimized hyperparamters

    '''

    # combining data for cross validation
    Xtrain=np.concatenate((Xtrain,Xvalidate),axis=0)
    ytrain=np.concatenate((ytrain,yvalidate),axis=0)

    train_scores = dict()
    parameters_selected=dict()

    # looping over classifiers
    # using grid and randomized search
    for classifier_name, classifier_obj in classifiers.items():
        x=parameters[classifier_name]['parameters']
        if len(x)==2:
            search_cv=RandomizedSearchCV(classifier_obj,x, cv=3)
        else:
            search_cv=GridSearchCV(classifier_obj,x, cv=3)
        search_cv.fit(Xtrain, ytrain)

        parameters_selected[classifier_name]=search_cv.best_estimator_

    return parameters_selected


def best_model(Xtrain, Xvalidate, Xtest,ytrain, yvalidate, ytest,classifiers):
    '''
    split data into test and train

    Parameters
    ----------
    Xtrain: DataFrame

        training data set

    Xvalidate: DataFrame

        validation data set

    Xtest: DataFrame

         test data set

    ytrain: DataFrame

        target for training data

    yvalidate: DataFrame

        target for validation data

    ytest: DataFrame

        target for test data

    classifers: dict

        dictionary of classifiers

    Returns
    -------
    df: DataFrame

        containing scores to compare classifers

    best_classifer: list

        list of best classifers for data

    '''

    train_scores = dict()
    validation_scores = dict()
    test_scores = dict()
    precision=dict()
    recall=dict()
    f1score=dict()

    # looping over classifiers
    # taking Xtrain_new and Xvalidate_new
    for classifier_name, classifier_obj in classifiers.items():
        classifier_obj.fit(Xtrain,ytrain)
        train_scores[classifier_name]=round(classifier_obj.score(Xtrain,ytrain)*100,2)
        validation_scores[classifier_name]=round(classifier_obj.score(Xvalidate,yvalidate)*100,2)
        test_scores[classifier_name]=round(classifier_obj.score(Xtest,ytest)*100,2)
        predict=classifier_obj.predict(Xtest)
        precision[classifier_name]= round((precision_score(ytest, predict))*100,2)
        recall[classifier_name] = round((recall_score(ytest, predict))*100,2)
        f1score[classifier_name]= round((f1_score(ytest, predict))*100,2)


    df = pd.DataFrame([train_scores,validation_scores,test_scores,precision,recall,f1score]).T
    df.rename(columns={0:'Training Accuracy',1:"Validation Accuracy",2:"Test Accuracy",
                       3:"Precision",4:"Recall",5:'F1 Score'},inplace=True)

    best_classifier=list(df.index[df['F1 Score']==max(df['F1 Score'])])

    return df,best_classifier

def generate_report(path,test_size,validate_size,random_state,response,classifiers,parameters):

    print("-------------------------------------------------------------------------------------")
    print("                              Step 1: Loading Data                                   ")
    print("-------------------------------------------------------------------------------------")
    print()
    print()
    data_df=load_data(path)

    print("-------------------------------------------------------------------------------------")
    print("                              Step 2: Outlier Removal                                ")
    print("-------------------------------------------------------------------------------------")
    print()
    print()
    data=outlier_removal(data_df)
    print("Outliers Removed",len(data_df)-len(data))
    print()
    print()

    print("-------------------------------------------------------------------------------------")
    print("                      Step 3: Descriptive Statistics                               ")
    print("-------------------------------------------------------------------------------------")
    summary_df,corr_df=data_summary(data)
    print()
    print()
    print("------------ Table 1: Descriptive Statistics of all numerical columns ---------------")
    print(summary_df)
    print("-------------------------------------------------------------------------------------")
    print()
    print()
    print("------------ Table 2: Numerical columns with significant correlation ----------------")
    print(corr_df)
    print("-------------------------------------------------------------------------------------")
    print()
    print()

    print("-------------------------------------------------------------------------------------")
    print("                                Step 4: Data Split                                   ")
    print("-------------------------------------------------------------------------------------")
    print()
    print()
    Xtrain, Xvalidate, Xtest, ytrain, yvalidate, ytest = data_split(data,response)
    print()
    print()

    print("-------------------------------------------------------------------------------------")
    print("                                Step 5: Fitting on Raw Data                          ")
    print("-------------------------------------------------------------------------------------")
    print()
    print()
    print("----------------------------------- Table 3: Scores ---------------------------------")
    scores=fit_classifers(Xtrain, Xvalidate, Xtest,ytrain, yvalidate, ytest,classifiers)
    print(scores)
    print("-------------------------------------------------------------------------------------")
    print()
    print()

    print("-------------------------------------------------------------------------------------")
    print("                          Step 6: Feature Transformation                             ")
    print("-------------------------------------------------------------------------------------")
    print()
    print()
    Xtrain_new,Xvalidate_new,Xtest_new = feature_transformation(Xtrain, Xvalidate, Xtest)
    print()
    print()

    print("-------------------------------------------------------------------------------------")
    print("                             Step 7: Feature Selection                               ")
    print("-------------------------------------------------------------------------------------")
    print()
    print()
    Xtrain_new,Xvalidate_new,Xtest_new=feature_selection(Xtrain_new,Xvalidate_new,Xtest_new,
                                                         ytrain, yvalidate, ytest)
    print()
    print()


    print("-------------------------------------------------------------------------------------")
    print("                Step 8: Fitting on Transformed and Selected Features                 ")
    print("-------------------------------------------------------------------------------------")
    print()
    print()
    print("----------------------------------- Table 4: Scores ---------------------------------")
    scores=fit_classifers(Xtrain_new,Xvalidate_new,Xtest_new,ytrain, yvalidate, ytest,classifiers)
    print(scores)
    print("-------------------------------------------------------------------------------------")
    print()
    print()
    print()


    print("-------------------------------------------------------------------------------------")
    print("                     Step 9: Hyperparamter Optimization                              ")
    print("-------------------------------------------------------------------------------------")
    print()
    print()
    optimized_classifiers=hyperparamter_optimization(Xtrain_new, Xvalidate_new,
                                                     ytrain, yvalidate,
                                                     classifiers, parameters)
    print()
    print()


    print("-------------------------------------------------------------------------------------")
    print(" Step 10: Fitting on Transformed and Selected Features with Optimized Hyperparamters ")
    print("-------------------------------------------------------------------------------------")
    print()
    print()
    print("----------------------------------- Table 5: Scores ---------------------------------")
    scores=fit_classifers(Xtrain_new,Xvalidate_new,Xtest_new,
                          ytrain, yvalidate, ytest,
                          optimized_classifiers)
    print(scores)
    print("-------------------------------------------------------------------------------------")
    print()
    print()
    print()

    print("-------------------------------------------------------------------------------------")
    print("                                 Step 11: Best Model                                 ")
    print("-------------------------------------------------------------------------------------")
    print()
    print()
    scores,best_classifier=best_model(Xtrain_new,Xvalidate_new,Xtest_new,
                      ytrain, yvalidate, ytest,
                      optimized_classifiers)
    print()
    print()
    print("Best Classifer for the data is/are:",best_classifier)
    print()
    print()
    print("-------------------------------- Table 5: Final Scores -------------------------------")
    print(scores)
    print("-------------------------------------------------------------------------------------")
    print()
    print()
