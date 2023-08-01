import streamlit as st
import pandas as pd

# Function to remove a column from the DataFrame and update the session state
def remove_column(df, columns):
    df = df.drop(columns=columns)
    st.session_state.df = df
    return df

# Title
st.title("CSV File Upload and Column Removal")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Load the CSV file and store it in the session state
if uploaded_file is not None:
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)

    st.write(st.session_state.df)

    # Dropdown to select a column
    column_to_remove = st.multiselect("Select a column to remove", st.session_state.df.columns)

    # Button to remove the selected column
    if st.button("Remove"):
        st.session_state.df = remove_column(st.session_state.df, column_to_remove)
        st.success(f"Column '{column_to_remove}' removed.")
        st.write(st.session_state.df)
else:
    st.warning("Please upload a CSV file.")
