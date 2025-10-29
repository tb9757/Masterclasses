import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import streamlit as st 

#page config
#set the layour for the dashboard - first step in doing streamlit
st.set_page_config(
    page_title="Titanic Explorer (No-ML)", #page on top of the tab
    page_icon="🚢",
    layout="wide"
)
st.title("Titanic Explorer")
st.caption(
    "Filter -> Summarise -> Visualise. Built with Streamlit + pandas + Matplotlib")

#Data Loading
@st.cache_data(show_spinner=False) 

#decorator (what does this do???) above a funtion, decorates the function. With cache data do not show the loading spinner

def load_titanic() -> pd.DataFrame: #this arrow then says that the return type will be a pandas dataframe
    df = sns.load_dataset("titanic").copy()
    # Canonic tidyups
    cat_cols = [
        'sex',
        'class',
        'embark_town',
        'who',
        'adult_male',
        'alone',
        'deck',
        'embarked',
        'alive'
    ]
    for c in df.columns:
        if c in cat_cols:
            df[c] = df[c].astype('category')
    # Rename some columns for clarity
    df = df.rename(columns={
        'embark_town': 'embark_town',  #why has spencer changed the names to their original names
        'pclass': 'pclass'
    })
    return df

@st.cache_data(show_spinner=False)
def describe_missing(df: pd.DataFrame) -> pd.DataFrame: #why is thre a colon in the parameters what does this mean
    miss = df.isna().mean().mul(100).round(2).rename('missing_%')  #what does mul do??
    miss = miss.reset_index()
    miss.columns = ['column', 'missing_%'] #what does this do?
    return miss.sort_values('missing_%', ascending= False) #what does this do?

df = load_titanic()  #why this line at the end

#make a sidebar with filters which can interact with the data

