
import streamlit as st
from app_core.configs import config

from app_core.utils.init_sessionvars import initSess
from app_core.modules import topicmodelling

task = st.sidebar.selectbox('Tasks', config._TASKS)
session = initSess()

if task == 'EDA':
    topicmodelling.run(session)
elif task == 'Regression':
    st.header('Stay tuned, more to follow!')
elif task == 'Classification':
    st.header('Stay tuned, more to follow!')
elif task == 'Clustering':
    st.header('Stay tuned, more to follow!')
else:
    st.header('Welcome to the application!')
    st.subheader('Select a task to proceed.')






