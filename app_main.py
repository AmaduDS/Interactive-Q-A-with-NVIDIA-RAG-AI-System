import streamlit as st
import os
import subprocess

# Store the process in session state to terminate if needed
if 'current_process' not in st.session_state:
    st.session_state.current_process = None

# Interface utilisateur principale pour choisir la langue
st.sidebar.title("Settings / Paramètres")

# Adding a neutral option "Select a language"
language_choice = st.sidebar.selectbox("Choose your language / Choisissez votre langue", 
                                       ["Select a language", "English", "Français"], key="language_select")

# Function to terminate a running process
def terminate_process():
    if st.session_state.current_process:
        st.session_state.current_process.terminate()
        st.session_state.current_process = None

# Language selection logic
if language_choice == "Select a language":
    st.write("Please select a language to continue.")
    terminate_process()  # Ensure no app is running if "Select a language" is chosen

elif language_choice == "English":
    st.write("Switching to English version...")
    terminate_process()  # Terminate any previous app
    st.session_state.current_process = subprocess.Popen(["streamlit", "run", "app_llama.py"])

elif language_choice == "Français":
    st.write("Bascule vers la version française...")
    terminate_process()  # Terminate any previous app
    st.session_state.current_process = subprocess.Popen(["streamlit", "run", "app_mistral.py"])
