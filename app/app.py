import streamlit as st

from RandomForest import rf_param_selector
from DecisionTree import dt_param_selector
from XGBoost import xgb_param_selector
from LogReg import lr_param_selector

from functions import *

st.set_page_config(page_title="Sopra Steria Project", layout="wide")

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

RANDOM_SEED = 1

st.title("**Interface projet Sopra Steria** ")

st.subheader("**Sujet**")
st.write("""

Mettre en évidence, à partir de ces données, une problématique Data Science d’intérêt de votre choix, présentant une valeur métier, et de la traiter.

 """)

st.subheader("**Description des données**")

st.write("""

Le dossier data/ contient deux fichiers : data.csv et socio_demo.csv. Les données sont tirées
de campagnes de marketing direct (démarchage téléphonique) d'une grande institution
bancaire, relatives à un produit de « dépôt à terme ». Un dépôt à terme est un dépôt
bancaire qui ne peut être retiré qu'à l'échéance d'un certain terme ou d'une certaine
période, en contrepartie d’un taux d’intérêt généralement plus élevé que pour un dépôt
classique (à vue).

 """)

st.header("**Importation des données** 💻")

with st.expander("Description des données:", expanded=False):
     st.write("""

- **DATE** (datetime) : Date du dernier contact 
- **AGE** (int): Âge du client en années
- **JOB** (string): Type de métier du client 
- **RELATION** (string): Statut marital du client 
- **EDUCATION** (string): Niveau d’éducation correspondant au diplôme le plus avancé obtenu par le client
- **DEFAUT**  (int): Indique si le client a déjà fait défaut par le passé 
- **BALANCE** (float): Solde du client 
- **PRET_IMMO** (string): Indique si le client a un prêt immobilier
- **PRET_PERSO** (string): Indique si le client a un prêt à la consommation
- **CONTACT** (string): Moyen de contact avec le client 
- **DUREE_CONTACT** (int): Durée du dernier échange avec le client en secondes
- **NB_CONTACT** (int): Nombre de contacts effectués avec le client durant cette campagne (y compris le dernier contact)
- **NB_J_DERNIER_CONTACT** (int): Nombre de jours écoulés après que le client a été contacté pour la dernière fois lors d’une campagne précédente (-1 si le client n’a jamais été contacté auparavant)
- **NB_CONTACT_DERNIERE_CAMPAGNE** (int): Nombre de contacts effectués avec ce client lors de la dernière campagne 
- **RESULTAT_DERNIERE_CAMPAGNE** (string): Résultat de la précédente campagne marketing 
- **STATUT** (string): Statut dans la campagne actuelle. 
                Souscrit : produit souscrit par le client
                Refus : produit refusé par le client
                En attente : en attente d’un retour ou d’une action du client
                Absent : le client n'a pas décroché après des sollicitations multiples pour cette campagne
     """)

uploaded_file = st.file_uploader("Importez le fichier csv.", "csv")

if uploaded_file:
    dataset = load_datas(uploaded_file, encoding="UTF-8")
    st.write(dataset)


st.header("**Data Visualisation** 🔎")
st.write("Vous pouvez visualiser les données et les corrélations entre elles ici. Chargez le dataset pour utiliser cette partie.")

st.subheader("**Statistiques**")
st.write("Ici, vous pouvez voir certaines statistiques des variables, comme la moyenne ou l'écart-type.")

if  uploaded_file:
    col1, _ , col3 = st.columns([3, 1, 3])

    numerical_cols, categorical_cols = find_numerical_categorical_cols(dataset)

    feature_selected = col1.selectbox("Nom de la variable", [None] + numerical_cols)
    if feature_selected:
        col3.write("Description de la variable")
        col3.write(dataset[feature_selected].describe())



st.subheader("**Graphiques**")
st.write("Vous pouvez visualiser les données et les corrélations entre elles ici. Chargez le dataset pour utiliser cette partie.")

if uploaded_file:
    distribution_plot(dataset)
    scatter_features(dataset)
    donut(dataset)
    time_series(dataset)
    #heatmap(dataset)


st.sidebar.image("ss_logo.png")

st.sidebar.header("Création du modèle de prédiction")

side_expander_split = st.sidebar.expander("Split du dataset", expanded=False)
test_size = side_expander_split.slider("taille du dataset de test (%)", 0, 100, 20)

models = {
    "Logistic regression":lr_param_selector,
    "Decision Tree":dt_param_selector,
    "Random Forest":rf_param_selector,
    "Xtreme Gradient Boosting":xgb_param_selector,
}

side_expander_train = st.sidebar.expander("Entrainez un modèle", expanded=True)
model_selected = side_expander_train.selectbox("Choisissez un moodèle", models.keys())

side_expender_tune = st.sidebar.expander("Modifiez les hyperparamètres", expanded=True)
with side_expender_tune:
    model = models[model_selected](RANDOM_SEED)

button = st.sidebar.button("Démarrez l'entrainement")


### Split en X et y et scale les données si le dataset a été uploadé
if uploaded_file:
    X, y = seperate_X_y(dataset)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size/100, RANDOM_SEED)
    X_train_scaled, X_test_scaled = scaling(X_train, X_test)


### Bas de page





st.header("**Validation du modèle** ✔️")
st.write("Vous devez d'abord créer un modèle dans le menu de gauche.")

if (uploaded_file) and button:#Executer les fonctions si le dataset a été uploadé et que le bouton a été activé
    model = create_model(model, X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    model_report(y_test, y_pred)

    col1, col3 = st.columns(2)
