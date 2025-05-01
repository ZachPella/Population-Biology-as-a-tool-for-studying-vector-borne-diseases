import streamlit as st
import Reed_Frost
import Lewis_Leslie_Roach
import Lewis_Leslie_Mosquito
import Macdonald
import model_assistant  # Import the assistant module

st.set_page_config(page_title="Population Biology Models", layout="wide")
st.sidebar.title("🧬 Zach's Disease Modeling Suite")

# Add the assistant option to your radio selection
app = st.sidebar.radio("Choose a model or assistant:", [
    "Reed-Frost",
    "Leslie Matrix - Roach",
    "Leslie Matrix - Mosquito",
    "Macdonald Model",
    "Model Assistant"  # New option for the chatbot
])

# Run the appropriate module based on selection
if app == "Reed-Frost":
    Reed_Frost.run()
elif app == "Leslie Matrix - Roach":
    Lewis_Leslie_Roach.run()
elif app == "Leslie Matrix - Mosquito":
    Lewis_Leslie_Mosquito.run()
elif app == "Macdonald Model":
    Macdonald.run()
elif app == "Model Assistant":
    # Display a small header for the assistant page
    st.header("Population Biology Model Assistant")
    st.markdown("""
    Ask questions about population biology models, epidemiology concepts, 
    or get help with interpreting model results. This assistant has internet 
    access to provide up-to-date information.
    """)
    
    # Add a divider
    st.divider()
    
    # Run the assistant module
    model_assistant.run()
