#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
#import xgboost 
from sys import exit


def transform_data(df,col_omit):
    """
    Convert categorical,bool data into 
    numbers and take care of 
    null and NaN values
    """

    for col in df.columns:
        if col in col_omit:
            #print col, "not changed"
            continue
        if df[col].dtype =="object" or df[col].dtype=="bool":
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes

    df.fillna(-1)
    df[df.isnull()] = -1

    return df

def variance_cutoff(X,cutoff=0.8):
    """
    Set variance cutoff for variables
    """
    sel = VarianceThreshold(threshold=(cutoff * (1 - cutoff)))
    X = sel.fit_transform(X)
    return X


def get_X_y(df):
    """
    get features(X) and target(y)
    as separate variables
    """
    cols = df.columns
    try:
        xcols = cols.drop(['ID','target'])
    except:
        xcols = cols.drop('ID')

    X = df[xcols].values
    try:
        y = df['target'].values
    except:
        y = []
    return X,y


if __name__=="__main__":
    
    train_files = []
    test_files = []

    fsub = open("submission.csv","w")
    fsub.write("ID,target\n")
    fscore = open("score.dat","w")
    
    for i in range(0,10):
        test_files.append("test."+str(i)+".csv")
        train_files.append("train."+str(i)+".csv")
        
    GB = GradientBoostingClassifier()

##    # variance cut-off    
##    cutoff = 0.8
##    sel = VarianceThreshold(threshold=(cutoff * (1 - cutoff)))

    all_y = [] # all y values for submission
    ID = np.array([])  # Index

    # load data file by file for ease of handling
    for i in range(0,10):
        print "reading ", train_files[i]
        df1= pd.read_csv(train_files[i],low_memory=False)
        df1 = transform_data(df1,['ID','target'])
        X,y = get_X_y(df1)
        
###        # variance cutoff
###        print X.shape
###        X = sel.fit_transform(X)
#3#        print X.shape

        # model fitting
        GB.fit(X,y)

        # Validation on other files except the one used
        # to build the model
        for j in range(0,10):
            if i == j: continue
            df= pd.read_csv(files[j],low_memory=False)
            df = transform_data(df,['ID','target'])
            X,y = get_X_y(df)
            #X = sel.transform(X)
    
            score = GB.score(X,y)
            wdata = "{:4d} {:4d} {:8.4f}\n".format(i,j,score)
            print j,i, wdata[:-1]
            fscore.write(wdata)

        # Prediction
        TARGET = np.array([])
        counter = 0
        for file_name in test_files:
            counter += 1
            print "reading file", file_name
            df= pd.read_csv(file_name, low_memory=False)
            df = transform_data(df,['ID','target'])
            X,y = get_X_y(df)
            #X = sel.transform(X)
            y = GB.predict_proba(X)
            TARGET = np.concatenate((TARGET,y[:,0]))
            if i==1:
                ID = np.concatenate((ID,df['ID'].values))
        all_y.append(TARGET)

        
    fscore.close()

    all_y = np.array(all_y)
    y_mean = np.mean(all_y, axis=0)
    for id,target in zip(ID,y_mean):
        wdata = "{},{}\n".format(id,target)
        fsub.write(wdata)

    fsub.close()
    
