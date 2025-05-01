# model_assistant.py - Claude-powered assistant for population biology models
import streamlit as st
import anthropic
import os
from datetime import datetime
import random

def local_css():
    """Define custom CSS styles"""
    st.markdown("""
    <style>
    /* Main page styling */
    .main {
        padding: 30px;
    }
    
    /* Header styling */
    .title-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 40px;
        gap: 15px;
    }
    
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        color: #026302;
        font-weight: 600;
        font-size: 2.5rem;
        margin: 0;
        text-align: center;
    }
    
    .robot-icon {
        font-size: 3rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #d31010;
        font-size: 1.9rem !important;
        margin: 30px 0;
        text-align: center;
        font-weight: 500;
    }
    
    /* Question label */
    .question-label {
        font-size: 2rem !important;
        margin-bottom: 20px;  /* Increased space below label */
        color: #FFF;
    }
    
    /* Simple search bar that doesn't get cut off */
    div[data-testid="stTextInput"] {
        margin-bottom: 60px !important; 
    }
    
    div[data-testid="stTextInput"] input {
        height: 45px !important;
        font-size: 1.2rem !important;
        padding: 10px !important;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 60px;
        text-align: center;
        color: #FFF;
        font-size: 1.5rem;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #f0f2f6 !important;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #e6eeff !important;
    }
    
    /* Removing padding for cleaner look */
    .css-18e3th9 {
        padding-top: 0;
    }
    
    /* Simple button styling */
    .stButton button {
        border-radius: 4px;
        padding: 8px 16px;
        background-color: #0066cc;
        color: white;
        border: none;
    }
    
    .stButton button:hover {
        background-color: #0052a3;
    }
    </style>
    """, unsafe_allow_html=True)

def get_fallback_response(prompt):
    """Generate a fallback response based on the query without using API"""
    prompt = prompt.lower()
    
    # Reed-Frost model responses
    if "reed" in prompt and "frost" in prompt:
        if "c" in prompt or "contact" in prompt:
            return """
In the Reed-Frost model, 'c' represents the adequate contact rate or transmission parameter.

The adequate contact rate 'c' is the probability that a susceptible individual will become infected after contact with an infectious individual. It combines both:

1. The probability of contact between susceptible and infectious individuals
2. The probability that such contact will lead to infection

This parameter is crucial in determining how quickly an epidemic spreads through a population. A higher value of 'c' indicates more efficient disease transmission.

In the mathematical formulation of the Reed-Frost model, 'c' is used to calculate the number of new infections in each time step using the binomial probability distribution.
"""
        
        elif "formula" in prompt or "equation" in prompt or "math" in prompt:
            return """
The Reed-Frost model can be mathematically expressed with these key equations:

1. Expected number of new cases: E[C(t+1)] = S(t) × (1 - (1-p)^I(t))

Where:
- C(t+1) = New cases at time t+1
- S(t) = Number of susceptible individuals at time t
- I(t) = Number of infectious individuals at time t
- p = Probability of effective contact (sometimes denoted as 'c')

2. Transition equations:
   - S(t+1) = S(t) - C(t+1)
   - I(t+1) = C(t+1)
   - R(t+1) = R(t) + I(t)

Where R(t) represents recovered/removed individuals at time t.

This model assumes a homogeneously mixing population with discrete time periods.
"""
        
        else:
            return """
The Reed-Frost model is a mathematical model used in epidemiology to predict the spread of infectious diseases in a closed population. It was developed by Lowell Reed and Wade Hampton Frost in the 1920s.

Key characteristics of the Reed-Frost model:

1. Discrete time intervals: The model progresses in distinct time steps or "generations" of infection.

2. Closed population: No births, deaths, or migration occur during the epidemic.

3. Homogeneous mixing: All individuals have equal probability of coming into contact with each other.

4. Chain binomial structure: The number of new infections follows a binomial probability distribution.

5. SIR framework: Individuals are classified as Susceptible, Infectious, or Recovered/Removed.

6. No latent period: Infected individuals become infectious immediately in the next time step.

7. Fixed infectious period: Individuals remain infectious for exactly one time unit.

The model is particularly useful for studying disease outbreaks in small, well-defined populations like schools, military units, or contained communities.
"""
    
    # Leslie Matrix responses
    elif "leslie" in prompt and "matrix" in prompt:
        if "approach" in prompt or "model" in prompt:
            return """
The Leslie Matrix approach for modeling population dynamics is a mathematical framework that incorporates age structure to predict population growth over time.

Key aspects of the Leslie Matrix approach:

1. Age-structured population: The population is divided into discrete age classes or cohorts.

2. Mathematical representation: Uses a square matrix where:
   - The first row contains fertility rates for each age class
   - The subdiagonal contains survival probabilities between age classes
   - All other elements are zero

3. Discrete time steps: The model advances in fixed time intervals (usually years).

4. Vector representation: The population at each time step is represented as a vector, with each element corresponding to the number of individuals in a specific age class.

5. Population projection: Future population structure is calculated by multiplying the Leslie matrix by the current population vector.

6. Eigenvalue analysis: The dominant eigenvalue of the Leslie matrix represents the long-term population growth rate.

This approach is particularly valuable for species where reproductive rates vary significantly with age, allowing for more accurate population forecasting than simpler models.
"""
        
        else:
            return """
The Leslie Matrix model is a discrete, age-structured population model used in ecology and demography. Developed by P.H. Leslie in 1945, it's a powerful tool for predicting how populations change over time when age-specific birth and death rates are known.

The Leslie Matrix (L) has this structure:
- First row: Contains fertility rates (Fᵢ) for each age class
- Subdiagonal: Contains survival probabilities (Pᵢ) between consecutive age classes
- All other elements: Zero

For example, a Leslie Matrix for a population with three age classes looks like:
[ F₁  F₂  F₃ ]
[ P₁   0   0 ]
[  0  P₂   0 ]

The population at the next time step is calculated by multiplying the Leslie Matrix by the current population vector:
n(t+1) = L × n(t)

Key assumptions:
- Fixed time intervals
- No immigration or emigration
- Fertility and survival rates remain constant
- No density-dependent effects

This model is especially useful for:
- Wildlife management
- Conservation of endangered species
- Human demography
- Pest control strategies
"""
    
    # General response for population biology
    else:
        # List of general responses about population biology
        general_responses = [
            """
Population biology models are mathematical frameworks that help us understand how populations change over time and respond to various factors.

Key types of population models include:

1. Exponential growth models: N(t) = N₀eʳᵗ - Simple models assuming unlimited resources
2. Logistic growth models: dN/dt = rN(1-N/K) - Incorporate carrying capacity
3. Structured population models: Account for age, stage, or spatial structure
4. Metapopulation models: Focus on connected populations with migration
5. Epidemic models: Describe disease spread within populations
6. Predator-prey models: Capture interacting species dynamics
7. Competition models: Describe resource competition between species

These models help researchers understand population dynamics, predict future population states, and inform conservation, disease control, and resource management strategies.
""",
            
            """
Population biology models serve several important purposes in research and application:

1. Prediction: Forecasting future population sizes under various scenarios
2. Understanding: Revealing the mechanisms driving population changes
3. Management: Informing conservation, harvesting, and pest control decisions
4. Hypothesis testing: Evaluating competing theories about population regulation
5. Risk assessment: Estimating extinction risks for threatened species
6. Disease control: Planning effective interventions for epidemic management
7. Evolutionary insights: Exploring how selection pressures affect populations over time

These models range from simple mathematical expressions to complex computational simulations, depending on the research questions and available data.
""",
            
            """
When studying population biology models, it's important to understand these fundamental concepts:

1. Growth rate (r): The per capita rate of increase of a population
2. Carrying capacity (K): The maximum population size an environment can sustain
3. Density dependence: How population growth rates change with population density
4. Stochasticity: Random variation in population processes
5. Time lags: Delayed effects in population responses
6. Spatial structure: How population distribution affects dynamics
7. Life history traits: Characteristics like survival and reproduction rates
8. Model parameters: Quantities that must be estimated from data
9. Equilibrium: Population states that remain stable over time
10. Sensitivity analysis: Determining which parameters most affect model outcomes

Understanding these concepts helps in building, interpreting, and applying population models effectively.
"""
        ]
        
        # Return a random general response
        return random.choice(general_responses)

def run():
    """Main function to run the assistant interface"""
    
    # Apply styling
    local_css()
    
    # Initialize session state values
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Add this to track the last prompt and prevent looping
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = None
    
    # Title with robot emoji
    # Title with robot emoji - enhanced with gradient
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <h1 style="font-family: 'Helvetica Neue', sans-serif; 
                 font-size: 3.5rem;
                 font-weight: 700;
                 background-image: linear-gradient(45deg, #9553E9, #6F42C1);
                 background-size: 100%;
                 background-clip: text;
                 -webkit-background-clip: text;
                 -moz-background-clip: text;
                 -webkit-text-fill-color: transparent;
                 -moz-text-fill-color: transparent;
                 text-fill-color: transparent;
                 margin: 20px 0 10px 0;
                 padding: 0;
                 letter-spacing: 2px;
                 display: inline-block;">
            🤖 Population Biology Model Assistant
        </h1>
        <br>
        <h4 style="font-family: 'Helvetica Neue', sans-serif;
               font-size: 1.2rem;
               color: #B399D4;
               margin-top: 0;
               margin-bottom: 25px;
               font-weight: 400;">
            Ask questions about population biology models:
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Subtitle
    st.markdown('<p class="subtitle">Ask questions about population biology models:</p>', unsafe_allow_html=True)
    
    # Configure sidebar for model selection
    with st.sidebar:
        st.subheader("Assistant Settings")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.last_prompt = None
            st.rerun()
    
    # System message - moved from sidebar to be hidden
    system_message = """You are a specialized assistant for population biology and epidemiological modeling.
            
You help students and researchers understand concepts related to:
- Reed-Frost models
- Leslie Matrix models
- Macdonald models
- SIR/SEIR models
- Vector-borne disease dynamics
- Population biology principles

Provide educational, accurate responses with relevant equations, explanations, and references. 
Use your internet access to find up-to-date information when needed.
When explaining mathematical concepts, be thorough but clear."""
    
    # Main container for question input
    st.markdown('<p class="question-label">Your Question:</p>', unsafe_allow_html=True)
    
    # User input field - simpler styling to avoid cutoff
    user_question = st.text_input("", 
                                label_visibility="collapsed", 
                                placeholder="Type your question about population biology models...",
                                key="user_question_input")
    
    # If user has entered a question and it's not a repeat
    if user_question and st.session_state.get("last_prompt") != user_question:
        st.session_state.last_prompt = user_question  # Prevent duplicate resend
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Display assistant response with a spinner
        with st.chat_message("assistant"):
            with st.spinner("Researching population biology concepts..."):
                # Generate fallback response directly (no API call needed)
                response = get_fallback_response(user_question)
                
                # Display response
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Use rerun to refresh interface
        st.rerun()
    
    # Display chat history if there are messages
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Footer with attribution
    st.markdown("""
    <div class="footer">
        Developed by Zach Pella
    </div>
    """, unsafe_allow_html=True)
