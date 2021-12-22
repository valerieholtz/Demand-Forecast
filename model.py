#!/usr/bin/env python
# coding: utf-8


""" Ridge model to predict Bike count """

import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge


pwd = os.getcwd()
df = pd.read_csv(pwd + '/train.csv', sep=",")


def create_dt_features(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    df['month'] = pd.DatetimeIndex(df['datetime']).month
    df['hour'] = pd.DatetimeIndex(df['datetime']).hour
    
    df.set_index(df['datetime'],inplace=True)
    df.drop('datetime', axis=1,inplace=True)
    return df


def train_model(df):
    
    print('TRAINING MODEL')
    
    #pick features
    y = df["count"]
    X = df[['workingday', 'weather', 'temp', 'month', 'hour']]

    model = make_pipeline(PolynomialFeatures(degree=2),Ridge())
    model.fit(X,y)
    return model


if __name__ == '__main__':
    create_dt_features(df)
    train_model(df)
    
    
