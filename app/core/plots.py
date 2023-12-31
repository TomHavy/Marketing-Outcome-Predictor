import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


long_col=["job","relation","Month_year","idx_prix_conso","idx_conf_conso","date"]
socio_eco_col=["tx_var_emploi",	"idx_prix_conso",	"idx_conf_conso"]        

st.cache_data
def distribution_plot(dataset):
    st.write('**Variables distribution**')

    open = st.checkbox("Display", key=2)

    if open: 

        select = st.selectbox("Choose a varible", dataset.columns,index=0)
        
        fig = px.histogram(dataset, x=select) 

        if select in long_col: 
            plt.xticks(rotation=90)

        st.plotly_chart(fig, use_container_width=True)


st.cache_data
def scatter_features(dataset):
    st.write('**Scatterplot**')

    open = st.checkbox("Display", key=4)

    if open:
        selected_x_var = st.selectbox('Choose x variable', dataset.columns)
        selected_y_var = st.selectbox('Choose y variable', dataset.columns, index= len(dataset.columns)-1)
        fig = px.scatter(dataset, x = dataset[selected_x_var], y = dataset[selected_y_var])
        st.plotly_chart(fig, use_container_width=True)

st.cache_data
def donut(dataset):

    souscrit= dataset[dataset['statut'] == 'Souscrit']
    souscrit_60_100=souscrit[(souscrit['age_group'] == '60 to 80') | (souscrit['age_group'] == '80 to 100')]
    
    cols_sub=["relation","education","pret_perso","pret_immo"]

    st.write('**Pie chart**')

    open = st.checkbox("Display", key=5)
    if open:
        select_donut = st.selectbox('Choose the variable to plot', cols_sub)

        counts_cols_sub = souscrit_60_100.groupby(select_donut)['date'].count()

        fig = px.pie(counts_cols_sub, values=counts_cols_sub,names=counts_cols_sub.index,hole=.5)
        st.plotly_chart(fig, use_container_width=True)

st.cache_data
def time_series(dataset):

    subscribers_per_month = dataset[dataset['statut'] == 'Souscrit'].groupby(['Month_year'])['statut'].count()

    st.write('**Time series**')

    open = st.checkbox("Display", key=6)
    if open:
        select = st.selectbox('Choose an economic indicator to compare', socio_eco_col)
        
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("#### Number of subscriptions over time")
            fig=px.line(subscribers_per_month, x=subscribers_per_month.index, y=subscribers_per_month.values)
            st.plotly_chart(fig)
        
        with fig_col2:
            st.markdown("#### Evolution of the economic indicators")
            fig2=px.line(dataset, x='date', y=select)
            st.plotly_chart(fig2)

# st.cache_data
# def heatmap(dataset):
#     numerical_cols, categorical_cols = find_numerical_categorical_cols(dataset)

#     # Correlation
#     df_corr = dataset.corr().round(1)  
#     # Mask to matrix
#     mask = np.zeros_like(df_corr, dtype=bool)
#     mask[np.triu_indices_from(mask)] = True
#     # Viz
#     df_corr_viz = df_corr.mask(mask).dropna(how='all').dropna('columns', how='all')
#     fig = px.imshow(df_corr_viz, text_auto=True)

#     st.plotly_chart(fig, use_container_width=True)

