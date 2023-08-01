import matplotlib.pyplot as plt
import streamlit as st

from app_core.utils.init_sessionvars import initSessVarsTopicMod
from projects.topic_extraction.script import TopicModelling
from pandas.api.types import is_string_dtype
from app_core.utils import helper


def run(session):

    st.header('Exploratory data analysis!')
    st.sidebar.header("Configure")

    initSessVarsTopicMod(session)

    if session.upload is not None:

        session.dataframe = helper._createDataFrame(session.upload) if not session.dataModified else session.dataframe

        if st.sidebar.button('Reset App'):
            session.clear()
            st._rerun()

        if st.checkbox('Preview Uploaded Data'):
            st.dataframe(helper._dataPreview(session.dataframe, rows=10))

        #TODO : Implement multiple model usage
        #st.sidebar.selectbox('Choose Model', config._MODELS.keys())
        session.selectedCols = st.sidebar.multiselect('Drop Columns(Multiple selections allowed)', session.dataframe.columns)

        if st.sidebar.button(label="Remove Columns"):
            session.dataModified = True
            session.dataframe = session.dataframe.drop(session.selectedCols, axis = 1)

        # TODO : functionality to label barplot axes
        value = st.sidebar.selectbox('Select column to plot Hist or Wordcloud.', session.dataframe.columns)
        cols = st.columns(2)
        with cols[0]:
            if st.button('Histogram') and value:
                if session.dataframe[value].nunique() <= 100:
                    count = session.dataframe.groupby(by=value).size()
                    #TODO : Add axes labels
                    st.pyplot(count.plot(kind='bar').figure)
                else:
                    st.error('Too many values to plot. Choose another column!')

        with cols[1]:
            if st.button('Wordcloud') and value:
                if is_string_dtype(session.dataframe[value]):
                    try:
                        img = TopicModelling._createWordCloud(session.dataframe[value])
                        fig, ax = plt.subplots(figsize=(12, 8))
                        ax.imshow(img)
                        plt.axis("off")
                        st.pyplot(fig)
                    except:
                        st.error('Incorrect or missing data. Choose another column!')
                else:
                    st.error('Columns with "str" datatype allowed. Choose another column!')


    else:
        st.sidebar.warning('Upload csv data.')