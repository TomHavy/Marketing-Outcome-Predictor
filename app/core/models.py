import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def dt_param_selector(seed):

    max_depth = st.number_input("max_depth", 1, 50, 5, 1)
    min_samples_split = st.number_input("min_samples_split", 1, 20, 2, 1)
    max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"])

    params = {
        #"criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
    }

    model = DecisionTreeClassifier(**params)
    return model

def lr_param_selector(seed):
    params = {

    }
    model = LogisticRegression(**params)
    return model

def rf_param_selector(seed):

    criterion = st.selectbox("criterion", ["gini", "entropy"])
    n_estimators = st.number_input("n_estimators", 50, 300, 100, 10)
    max_depth = st.number_input("max_depth", 1, 50, 5, 1)
    min_samples_split = st.number_input("min_samples_split", 1, 20, 2, 1)
    max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"])

    params = {
        #"criterion": criterion,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "n_jobs": -1,
    }

    model = RandomForestClassifier(**params)
    return model

def xgb_param_selector(seed):
    n_estimators = st.number_input("n_estimators", 100, 1000, 100, 50)
    learning_rate = st.number_input("learning_rate (%)", 1, 100, 10, 1)
    max_depth = st.number_input("max_depth", 1, 10, 3, 1)

    params = {
        "n_estimators": n_estimators,
        "learning_rate":learning_rate/100,
        "max_depth": max_depth,
    }
    
    model = XGBClassifier(**params)
    return model
