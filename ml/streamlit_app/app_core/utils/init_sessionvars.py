
import streamlit as st

@st.cache_resource
def initSess():
    return st.session_state

def initSessVarsTopicMod(session):

    if 'dataModified' not in session.keys():
        session.dataModified = False

    if 'dataFrame' not in session.keys():
        session.dataFrame = None

    if 'upload' not in session.keys():
        session.upload = None

    if 'uploader' not in session.keys():
        session.uploader = st.empty()

    if session.upload is None:
        session.upload = session.uploader.file_uploader(label='Upload', type='csv')