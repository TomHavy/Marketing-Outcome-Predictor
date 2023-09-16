import streamlit as st

from core.plots import( 
    distribution_plot,
    scatter_features,
    donut,
    time_series,
)
from core.models import(
    rf_param_selector,
    dt_param_selector,
    xgb_param_selector,
    lr_param_selector
)
from core.functions import(
    load_datas,
    find_numerical_categorical_cols,
    model_report,
    create_model,
    scaling,
    split_dataset,
    seperate_X_y,    
)

st.set_page_config(page_title="Deposit Subscription Predictor", layout="wide", page_icon='🤖')

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


hide_streamlit_style = """
<style>
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("**Deposit Subscription Predictor** ")

st.header("**🚀 Project Overview**")
st.write("""

The Deposit Subscription Predictor is a data science and machine learning project that offers a comprehensive solution to a significant challenge faced by a major financial institution. It centers around a dataset derived from direct marketing campaigns, specifically telephone-based outreach, conducted by the bank to promote a "term deposit" product. Term deposits are financial instruments that can only be withdrawn upon reaching a specified term or maturity date, offering typically higher interest rates compared to regular demand deposits.
 """)

st.header("**💻 Data Overview**")

st.write("""

The data/ folder contains two files: data.csv and socio_demo.csv. The data are taken from
from the direct marketing campaigns (telephone canvassing) of a major banking
for a "term deposit" product. A term deposit is a bank deposit
which can only be withdrawn at the end of a certain term or period, in return for a
period, in return for an interest rate that is generally higher than that of a conventional (sight) deposit.
(sight) deposit.
         
 """)


dataset = load_datas(path="..\data\data_cleaned.csv", encoding="UTF-8")
st.write(dataset)

with st.expander("Column Description (english):", expanded=False):
     st.write("""

- **DATE** (datetime) : Date of last contact 
- **AGE** (int): Customer's age in years
- **JOB** (string): Customer's job type 
- **RELATIONSHIP** (string): Customer's marital status 
- **EDUCATION** (string): Level of education corresponding to the most advanced diploma obtained by the customer.
- **DEFAULT** (int): Indicates whether the customer has defaulted in the past. 
- **BALANCE** (float): Customer's balance 
- **PRET_IMMO** (string): Indicates whether the customer has a mortgage.
- **PRET_PERSO** (string): Indicates whether the customer has a consumer loan.
- **CONTACT** (string): Means of contacting the customer 
- **DUREE_CONTACT** (int): Duration of last exchange with customer in seconds
- **NB_CONTACT** (int): Number of contacts made with the customer during this campaign (including the last contact).
- **NB_LAST_CONTACT** (int): Number of days since the customer was last contacted in a previous campaign (-1 if the customer has never been contacted before).
- **NB_CONTACT_LAST_CAMPAGNE** (int): Number of contacts made with this customer during the last campaign. 
- **RESULT_LAST_CAMPAGNE** (string): Result of previous marketing campaign 
- **STATUS** (string): Status in the current campaign. 
                Subscribed: product subscribed by customer
                Refused: product refused by customer
                Waiting: waiting for feedback or action from the customer.
                Absent: customer has not picked up after multiple solicitations for this campaign.
     """)


st.header("**📊 Data Visualization**")
st.write("Explore the dataset and discover insightful correlations between the variables.")

st.subheader("**Statistics**")
st.write("In this section, you can access a range of statistical insights for the variables, including metrics like the mean and standard deviation.")

col1, _ , col3 = st.columns([3, 1, 3])

numerical_cols, categorical_cols = find_numerical_categorical_cols(dataset)

feature_selected = col1.selectbox("Variable name", [None] + numerical_cols)
if feature_selected:
    col3.write("Description of the variable")
    col3.write(dataset[feature_selected].describe())

st.subheader("**Plots**")
st.write("Explore the dataset and discover insightful correlations between the variables.")

distribution_plot(dataset)
scatter_features(dataset)
donut(dataset)
time_series(dataset)
#heatmap(dataset)

st.sidebar.header("Creating the prediction model")

side_expander_split = st.sidebar.expander("Dataset split", expanded=False)
test_size = side_expander_split.slider("test dataset size (%)", 0, 100, 20)

models = {
    "Logistic regression":lr_param_selector,
    "Decision Tree":dt_param_selector,
    "Random Forest":rf_param_selector,
    "Xtreme Gradient Boosting":xgb_param_selector,
}

side_expander_train = st.sidebar.expander("Train model", expanded=True)
model_selected = side_expander_train.selectbox("Choose a model", models.keys())

RANDOM_SEED = 1

side_expender_tune = st.sidebar.expander("Modify hyperparameters", expanded=True)
with side_expender_tune:
    model = models[model_selected](RANDOM_SEED)

button = st.sidebar.button("Start training")

X, y = seperate_X_y(dataset)
X_train, X_test, y_train, y_test = split_dataset(X, y, test_size/100, RANDOM_SEED)
X_train_scaled, X_test_scaled = scaling(X_train, X_test)




st.header("**✅ Model validation** ")
st.write("You must first create a template in the left-hand menu.")

if  button:
    with st.spinner('Training...'):
        model = create_model(model, X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    model_report(y_test, y_pred)

    col1, col3 = st.columns(2)
