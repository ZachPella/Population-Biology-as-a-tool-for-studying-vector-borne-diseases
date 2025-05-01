import streamlit as st
import requests
import json

def add_chat_interface():
    """Add the chatbot interface to the Streamlit sidebar"""
    st.sidebar.title("📚 Model Assistant")
    st.sidebar.write("Ask me about population biology concepts!")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.sidebar.chat_message(message["role"]):
            st.sidebar.markdown(message["content"])
    
    # Suggestions for common questions
    if not st.session_state.messages:
        st.sidebar.markdown("### Try asking:")
        cols = st.sidebar.columns(2)
        
        if cols[0].button("Leslie Matrix basics"):
            user_query = "What is the Leslie Matrix model and how does it work?"
            st.session_state.messages.append({"role": "user", "content": user_query})
            _handle_query(user_query)
            
        if cols[1].button("Vectorial capacity"):
            user_query = "Explain Macdonald's vectorial capacity model"
            st.session_state.messages.append({"role": "user", "content": user_query})
            _handle_query(user_query)
            
        if cols[0].button("Survival impact in VC"):
            user_query = "Why does daily survival rate have a large impact in vectorial capacity?"
            st.session_state.messages.append({"role": "user", "content": user_query})
            _handle_query(user_query)
            
        if cols[1].button("Key model parameters"):
            user_query = "What are the key parameters in Leslie Matrix and Macdonald models?"
            st.session_state.messages.append({"role": "user", "content": user_query})
            _handle_query(user_query)
    
    # Chat input
    user_query = st.sidebar.text_input("Your question:", key="user_question")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.sidebar.chat_message("user"):
            st.sidebar.markdown(user_query)
        
        _handle_query(user_query)

def _handle_query(query):
    """Handle a user query, get a response, and update the chat"""
    with st.sidebar.spinner("Thinking..."):
        # In a real implementation, you would use a web search API
        # For now, we'll provide knowledge-based responses for common questions
        
        if "leslie matrix" in query.lower() or "age structured" in query.lower() or "age-structured" in query.lower():
            response = """
            **Leslie Matrix Model**
            
            The Leslie Matrix is a discrete, age-structured model used in population ecology to predict growth of populations. It was developed by P.H. Leslie in the 1940s.
            
            The model works by:
            - Dividing a population into age classes
            - Creating a square matrix with:
              - Fecundity values in the first row
              - Survival probabilities in the sub-diagonal
            - Multiplying this matrix by the population vector to project future population
            
            The basic equation is: 
            n(t+1) = M × n(t)
            
            Where:
            - n(t) is the population vector at time t
            - M is the Leslie Matrix
            
            This model helps predict both population size and age structure changes over time.
            
            Source: Leslie matrix - Wikipedia (https://en.wikipedia.org/wiki/Leslie_matrix)
            """
        elif "vectorial capacity" in query.lower() or "macdonald" in query.lower():
            response = """
            **Macdonald's Vectorial Capacity Model**
            
            Vectorial capacity is a key measure in vector-borne disease transmission, representing the number of potentially infectious bites arising from vectors that bite a single infectious host in one day.
            
            Macdonald's equation for vectorial capacity (V) is:
            V = ma²bp^n/-ln(p)
            
            Where:
            - m = vector density in relation to hosts
            - a = biting rate (humans bitten per mosquito per day)
            - b = vector competence (proportion of vectors developing infection)
            - p = daily vector survival rate
            - n = extrinsic incubation period (days)
            
            This model provides critical insights for vector control strategies.
            
            Source: Vectorial capacity and vector control (https://academic.oup.com/trstmh/article/110/2/107/2578714)
            """
        elif "daily survival" in query.lower() or "survival rate" in query.lower():
            response = """
            **Impact of Daily Survival Rate on Vectorial Capacity**
            
            The daily survival rate (p) has the strongest effect on vectorial capacity because:
            
            1. It appears twice in the equation: as p^n in the numerator and -ln(p) in the denominator
            2. Small changes in survival create large changes in vectorial capacity
            3. According to research, a 10% increase in survival can lead to a >200% increase in vectorial capacity
            
            This is why many vector control strategies focus on reducing adult vector survival through insecticides.
            
            Source: An Age-Structured Extension to the Vectorial Capacity Model (https://pmc.ncbi.nlm.nih.gov/articles/PMC3378582/)
            """
        elif "control" in query.lower() or "intervention" in query.lower():
            response = """
            **Vector Control Strategies Based on Models**
            
            Mathematical models like Leslie Matrix and Macdonald's vectorial capacity have important implications for disease control:
            
            For vector-borne diseases:
            - Reducing adult vector survival (p) has the largest impact
            - Vector control programs use this insight to focus on adult mosquito control
            - Insecticide-treated nets and indoor residual spraying target the adult stage
            
            For age-structured populations:
            - Leslie Matrix models help identify which life stages most affect population growth
            - Control strategies can target the most influential life stages
            
            Source: Vectorial capacity and vector control (https://pmc.ncbi.nlm.nih.gov/articles/PMC4731004/)
            """
        elif "parameter" in query.lower() or "sensitivity" in query.lower():
            response = """
            **Key Parameters in Population Biology Models**
            
            **Leslie Matrix key parameters:**
            - Fecundity values (first row): Average number of offspring per individual in each age class
            - Survival probabilities (sub-diagonal): Probability of surviving from one age class to the next
            
            **Vectorial Capacity key parameters:**
            - Vector:host ratio (m): Number of vectors per host
            - Biting rate (a): Appears squared in the equation (a²)
            - Daily survival rate (p): Has the strongest effect on vectorial capacity
            - Extrinsic incubation period (n): Time for pathogen development in vector
            - Vector competence (b): Proportion of vectors that develop infection
            
            Sensitivity analysis shows that survival rate has the most significant impact on vectorial capacity.
            
            Source: 7.3: Leslie Matrix Models - Biology LibreTexts (https://bio.libretexts.org/Courses/Gettysburg_College/02:_Principles_of_Ecology_-_Gettysburg_College_ES_211/07:_A_Quantitative_Approach_to_Population_Ecology/7.03:_Leslie_Matrix_Models)
            """
        else:
            response = """
            I can help answer questions about population biology models like the Leslie Matrix (for age-structured population dynamics) and Macdonald's Vectorial Capacity model (for vector-borne disease transmission).
            
            Some topics I can explain:
            - How these models work mathematically
            - Key parameters and their biological meaning
            - Applications to ecology and epidemiology
            - How these models inform control strategies
            
            Please feel free to ask a specific question about either model!
            """
        
        # Add response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display the response
        with st.sidebar.chat_message("assistant"):
            st.sidebar.markdown(response)
