import streamlit as st
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import plotly.figure_factory as ff

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,confusion_matrix,f1_score,accuracy_score,precision_score

def load_datas(path, encoding):
    dataset = pd.read_csv(path, encoding=encoding)
    return dataset

long_col=["job","relation","Month_year","idx_prix_conso","idx_conf_conso","date"]
socio_eco_col=["tx_var_emploi",	"idx_prix_conso",	"idx_conf_conso"]

def find_numerical_categorical_cols(dataset):
    #numerical_cols = [col for col in dataset.columns if dataset[col].dtype == float or dataset[col].dtype == int]
    #categorical_cols = [col for col in dataset.columns if dataset[col].dtype != float] #(object, int, str)]

    numerical_cols=["age","balance","nb_j_dernier_contact","nb_contact_derniere_campagne","idx_prix_conso","idx_conf_conso", "tx_var_emploi"]
    categorical_cols=['job','relation','education','defaut','pret_immo','pret_perso','duree_contact','nb_contact','resultat_derniere_campagne',"Month",'statut','Month_year','date','Year','age_group']
    return numerical_cols, categorical_cols

def _imbalance(X,y):
    oversample=SMOTE()
    undersample=RandomUnderSampler()
    steps=[["o",oversample],["u",undersample]]
    pipeline=Pipeline(steps=steps)

    X,y=pipeline.fit_resample(X,y)

    return X,y

st.cache_data
def seperate_X_y(dataset):
    X=dataset.drop(["date","Month_year","statut","duree_contact"],axis=1)

    X['defaut'] = dataset['defaut'].map({'Yes': 1, 'No': 0})
    X['pret_immo'] = dataset['pret_immo'].map({'Yes': 1, 'No': 0})
    X['pret_perso'] = dataset['pret_perso'].map({'Yes': 1, 'No': 0})
    X["resultat_derniere_campagne"]=X["resultat_derniere_campagne"].map({'Non Succes': '0', 'Succes': '1'})

    X=pd.get_dummies(X,columns=["job","relation","education","age_group"])

    y= dataset["statut"].replace({'Refus': '0', 'Souscrit': '1'}, regex=True)

    X,y=_imbalance(X,y)
    
    return X, y

st.cache_data
def split_dataset(X, y, test_size, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

    X_train["resultat_derniere_campagne"]=X_train["resultat_derniere_campagne"].astype(int)
    X_train["tx_var_emploi"]=X_train["tx_var_emploi"].astype(float)
    X_train["idx_prix_conso"]=X_train["idx_prix_conso"].astype(float)
    X_train["idx_conf_conso"]=X_train["idx_conf_conso"].astype(float)
    X_train["Month"]=X_train["Month"].astype(int)
    X_train["Year"]=X_train["Year"].astype(int)

    X_test["resultat_derniere_campagne"]=X_test["resultat_derniere_campagne"].astype(int)
    X_test["tx_var_emploi"]=X_test["tx_var_emploi"].astype(float)
    X_test["idx_prix_conso"]=X_test["idx_prix_conso"].astype(float)
    X_test["idx_conf_conso"]=X_test["idx_conf_conso"].astype(float)
    X_test["Month"]=X_test["Month"].astype(int)
    X_test["Year"]=X_test["Year"].astype(int)

    return X_train, X_test, y_train, y_test

st.cache_data
def scaling(X_train,X_test):

    min_max_scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(min_max_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(min_max_scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train_scaled,X_test_scaled

def create_model(model, X_train, y_train):
    model = model.fit(X_train, y_train)
    return model

st.cache_data
def _plot_confusion_matrix(y_test,y_pred):
 
    fig = ff.create_annotated_heatmap(confusion_matrix(y_test, y_pred), x=["0","1"], y=["0","1"], colorscale='Viridis')

    fig.update_layout(title_text='<b>Confusion matrix</b>')

    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    fig.update_layout(margin=dict(t=50, l=200))

    fig['data'][0]['showscale'] = True
    st.plotly_chart(fig)

def model_report(y_test, y_pred):
    st.write("Accuracy:", accuracy_score(y_test,y_pred))
    st.write("Precision:", precision_score(y_test,y_pred ,pos_label='1',average='binary'))
    st.write("Recall:", recall_score(y_test,y_pred ,pos_label='1',average='binary'))
    st.write("F1 score:", f1_score(y_test,y_pred ,pos_label='1',average='binary'))

    _plot_confusion_matrix(y_test,y_pred)
    