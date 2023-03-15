
import pandas as pd
import streamlit as st

@st.cache_data
def _createDataFrame(_fileUpload : bytes) -> pd.DataFrame:
    return pd.read_csv(_fileUpload)

@st.cache_data
def _dataPreview(dataframe: pd.DataFrame, rows: int = 10):
    try:
        return dataframe.head(rows)
    except:
        return dataframe.head()


