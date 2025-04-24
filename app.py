import streamlit as st
import Reed_Frost
import Lewis_Leslie_Roach
import Lewis_Leslie_Mosquito
import Macdonald

st.set_page_config(page_title="Population Biology Models", layout="wide")

st.sidebar.title("🧬 Zach's Disease Modeling Suite")
app = st.sidebar.radio("Choose a model:", [
    "Reed-Frost",
    "Leslie Matrix - Roach",
    "Leslie Matrix - Mosquito",
    "Macdonald Model"
])

if app == "Reed-Frost":
    Reed_Frost.run()
elif app == "Leslie Matrix - Roach":
    Lewis_Leslie_Roach.run()
elif app == "Leslie Matrix - Mosquito":
    Lewis_Leslie_Mosquito.run()
elif app == "Macdonald Model":
    Macdonald.run()
