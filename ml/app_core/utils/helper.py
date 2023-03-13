
import pandas as pd
import streamlit as st

@st.cache_data
def _createDataFrame(_fileUpload : bytes) -> pd.DataFrame:
    return pd.read_csv(_fileUpload)

def _dataPreview(dataFrame: pd.DataFrame, rows: int = 5):
    try:
        return dataFrame.head(rows)
    except:
        return dataFrame.head()

def _dropColumns(dataFrame : pd.DataFrame, colNames : list) -> pd.DataFrame:
    # User Input
    return dataFrame.drop(colNames, axis = 1)

def _removeCols(dataFrame, selectedCols):
    if selectedCols is not None and dataFrame is not None:
        st.session_state.dataFrame = _dropColumns(dataFrame, selectedCols)