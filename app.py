import streamlit as st
import Reed_Frost
import Lewis_Leslie_Roach
import Lewis_Leslie_Mosquito
import Macdonald

st.set_page_config(page_title="Epidemiological Models Dashboard", layout="wide")

st.sidebar.title("📊 Choose a Model")

model_choice = st.sidebar.radio("Select a simulation:", [
    "Reed-Frost",
    "Leslie Matrix - Roach",
    "Leslie Matrix - Mosquito",
    "Macdonald Model"
])

if model_choice == "Reed-Frost":
    Reed_Frost.run()
elif model_choice == "Leslie Matrix - Roach":
    Lewis_Leslie_Roach.run()
elif model_choice == "Leslie Matrix - Mosquito":
    Lewis_Leslie_Mosquito.run()
elif model_choice == "Macdonald Model":
    Macdonald.run()
