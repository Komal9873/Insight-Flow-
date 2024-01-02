import streamlit as st
import pandas as pd
import sweetviz as sv
import os
# import the libraray like streamlit numpy ,pandas and sweetviz as sv 


# Define a function for setting the background color and other styles
def set_style():
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(to bottom right, #d9e4ec, #f6f7fa);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_style()

if os.path.exists('dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Automated Machine Learning")
    choice = st.radio("Navigation", ["Upload", "EDA", "Modelling"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file)
        df.to_csv('dataset.csv', index=False)
        st.dataframe(df)

if choice == "EDA":
    st.title("Exploratory Data Analysis")
    st.subheader("Using SweetViz")
    
    if st.button("Generate EDA Report"):
        report = sv.analyze(df)
        report.show_html("eda_report.html")
        st.success("EDA report has been generated. You can download it below.")
        st.markdown("### [Download EDA Report](eda_report.html)")

if choice == "Modelling":
    st.title("Automated Machine Learning")

    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Automated Machine Learning'):
     
        from pycaret.regression import setup, compare_models, pull, save_model, load_model

       
        from tpot import TPOTRegressor

        
        setup(df, target=chosen_target)

        
        best_model_pycaret = compare_models()

       
        tpot = TPOTRegressor(verbosity=2, generations=5, population_size=20, random_state=42, config_dict='TPOT sparse')

        
        tpot.fit(df.drop(columns=[chosen_target]), df[chosen_target])

       
        best_pipeline = tpot.fitted_pipeline_

        
        st.write("Best Model (PyCaret):", best_model_pycaret)

        
        st.write("Best Pipeline (AutoML - TPOT):", best_pipeline)

        st.success("Automated Machine Learning completed.")

  
    pass