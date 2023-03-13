
import matplotlib.pyplot as plt
import streamlit as st

from app_core.utils.init_sessionvars import initSessVarsTopicMod
from topic_extraction.script import TopicModelling
from pandas.api.types import is_string_dtype
from app_core.utils import helper

def run(session):

    st.header('Exploratory data analysis!')
    st.sidebar.header("Configure")

    initSessVarsTopicMod(session)
    if session.upload is not None:

        if not session.dataModified:
            dataFrame = helper._createDataFrame(session.upload)
        else:
            dataFrame = session.dataFrame

        preview = st.checkbox('Preview Uploaded Data')
        if preview:
            st.dataframe(helper._dataPreview(dataFrame, rows=10))

        if st.sidebar.button('Reset App'):
            session.clear()
            st._rerun()

        #TODO : Implement multiple model usage
        #st.sidebar.selectbox('Choose Model', config._MODELS.keys())

        selectedCols = st.sidebar.multiselect('Drop Columns(Multiple selections allowed)', dataFrame.columns)

        if st.sidebar.button(label="Remove Columns", on_click=helper._removeCols, args=(dataFrame, selectedCols)):
            session.dataModified = True

        # TODO : functionality to label barplot axes
        value = st.sidebar.selectbox('Select column to plot Hist or Wordcloud.', dataFrame.columns)

        cols = st.columns(2)
        with cols[0]:
            plot = st.button('Histogram')
            if value and plot:
                if dataFrame[value].nunique() <= 100:
                    count = dataFrame.groupby(by=value).size()
                    #TODO : Add axes labels
                    st.pyplot(count.plot(kind='bar').figure)
                else:
                    st.error('Too many values to plot. Choose another column!')

        with cols[1]:
            wCloud = st.button('Wordcloud')
            if wCloud and value:
                if is_string_dtype(dataFrame[value]):
                    try:
                        img = TopicModelling._createWordCloud(dataFrame[value])
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