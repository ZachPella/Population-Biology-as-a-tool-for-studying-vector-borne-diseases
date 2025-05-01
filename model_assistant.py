# model_assistant.py - Claude-powered assistant for population biology models
import streamlit as st
import anthropic
import os
from datetime import datetime
import base64
from random import choice

def add_bg_from_url(url):
    """Add background image from URL"""
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url({url});
             background-attachment: fixed;
             background-size: cover;
             background-position: center;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def local_css():
    """Define custom CSS styles"""
    st.markdown("""
    <style>
    /* Customize chat messages */
    .stChatMessage {
        border-radius: 15px !important;
        padding: 8px 12px;
        margin-bottom: 10px;
    }
    
    /* Container styling */
    .main-container {
        background-color: rgba(17, 17, 40, 0.7);
        border-radius: 20px;
        padding: 25px;
        margin: 10px 0;
        border: 1px solid rgba(120, 120, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Header styling */
    .custom-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #E0E0FF;
        font-weight: 600;
        font-size: 2rem;
        margin-bottom: 5px;
        text-shadow: 0px 0px 10px rgba(100, 149, 237, 0.5);
    }
    
    /* Description text */
    .description-text {
        color: #C0C0FF;
        font-size: 1rem;
        font-weight: 300;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    
    /* Button styling */
    .stButton button {
        background-color: rgba(100, 149, 237, 0.7) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 15px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        background-color: rgba(100, 149, 237, 0.9) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Chat input styling */
    .stChatInputContainer {
        border-radius: 15px !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(120, 120, 255, 0.3) !important;
        padding: 8px !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: rgba(25, 25, 50, 0.9) !important;
    }
    
    /* Custom info box */
    .info-box {
        background-color: rgba(100, 149, 237, 0.2);
        border-left: 4px solid rgba(100, 149, 237, 0.7);
        padding: 10px 15px;
        border-radius: 0 10px 10px 0;
        margin: 20px 0;
        font-size: 0.9rem;
    }
    
    /* Footer text */
    .footer-text {
        color: #A0A0D0;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

def get_random_tip():
    """Return a random tip about population models"""
    tips = [
        "Reed-Frost models are useful for studying disease spread in small populations.",
        "Leslie Matrix models help understand age-structured population dynamics.",
        "The Macdonald model is essential for studying vector-borne diseases like malaria.",
        "SIR models divide the population into Susceptible, Infected, and Recovered groups.",
        "The basic reproduction number (R₀) indicates how contagious an infectious disease is.",
        "Leslie matrices can predict future population sizes based on age-specific fertility and survival rates.",
        "Vector-borne diseases require understanding both host and vector population dynamics.",
        "Population thresholds are critical points where disease dynamics change dramatically.",
    ]
    return choice(tips)

def run():
    """Main function to run the assistant interface"""
    
    # Set page config
    st.set_page_config(
        page_title="Population Biology Model Assistant",
        page_icon="🧬",
        layout="wide"
    )
    
    # Add background image
    add_bg_from_url("https://images.unsplash.com/photo-1557682250-27c44a49ea8d?q=80&w=2148&auto=format&fit=crop")
    
    # Apply custom CSS
    local_css()
    
    # Initialize session state for chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Container for header
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="custom-header">🧬 Population Biology Model Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description-text">Ask questions about population biology models, epidemiology concepts, or get help with interpreting model results. This assistant has internet access to provide up-to-date information.</p>', unsafe_allow_html=True)
    
    # Add a random tip
    st.markdown(f'<div class="info-box">💡 <b>Did you know?</b> {get_random_tip()}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Configure sidebar for the assistant settings
    with st.sidebar:
        st.markdown('<h3 style="color: #E0E0FF;">Assistant Settings</h3>', unsafe_allow_html=True)
        
        # API key input (can be stored in environment variables or secrets in production)
        api_key = st.text_input("Anthropic API Key", type="password", 
                              help="Enter your Anthropic API key to enable the assistant")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        
        # Model selection
        model_options = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
        selected_model = st.selectbox("Model", model_options, 
                                    help="Select the Claude model to use")
        
        # Customize system message
        system_message = st.text_area(
            "System Instructions", 
            value="""You are a specialized assistant for population biology and epidemiological modeling.
            
You help students and researchers understand concepts related to:
- Reed-Frost models
- Leslie Matrix models
- Macdonald models
- SIR/SEIR models
- Vector-borne disease dynamics
- Population biology principles

Provide educational, accurate responses with relevant equations, explanations, and references. 
Use your internet access to find up-to-date information when needed.
When explaining mathematical concepts, be thorough but clear.""",
            height=200,
            help="Customize the assistant's instructions"
        )
        
        # Web search toggle
        use_web_search = st.toggle("Enable Web Search", value=True,
                                  help="Allow the assistant to search the internet for information")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main content area with container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Function to generate response from Claude
    def get_claude_response(prompt, history, system_message, model, use_web_search):
        try:
            # Initialize the Anthropic client
            client = anthropic.Client(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            
            # Prepare messages for Claude
            messages = []
            
            # Add chat history
            for msg in history:
                if msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})
                else:
                    messages.append({"role": "user", "content": msg["content"]})
            
            # Add current user message
            messages.append({"role": "user", "content": prompt})
            
            # Set up tools for web search if enabled
            tools = None
            if use_web_search:
                tools = [
                    {
                        "name": "web_search",
                        "description": "Search the web for information about population biology and epidemiological models",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                ]
            
            # Get response from Claude
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                temperature=0.7,
                system=system_message,
                messages=messages,
                tools=tools
            )
            
            return response.content[0].text
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    # Handle user input
    if prompt := st.chat_input("Ask about population models, disease dynamics, or model interpretation..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response with a spinner
        with st.chat_message("assistant"):
            with st.spinner("Researching population biology concepts..."):
                # Check if API key is provided
                if not api_key:
                    response = "Please enter your Anthropic API key in the sidebar to continue."
                else:
                    # Generate response
                    response = get_claude_response(
                        prompt=prompt,
                        history=st.session_state.messages[:-1],  # Exclude the current message
                        system_message=system_message,
                        model=selected_model,
                        use_web_search=use_web_search
                    )
                
                # Display response
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # If no messages yet, show example questions
    if len(st.session_state.messages) == 0:
        st.markdown('<div style="color: #A0A0D0; padding: 20px; text-align: center;">', unsafe_allow_html=True)
        st.markdown("### Sample Questions to Ask:")
        
        example_questions = [
            "How does a Reed-Frost model work?",
            "Explain the Leslie Matrix approach for modeling population dynamics",
            "What factors influence the basic reproduction number (R₀) in vector-borne diseases?",
            "How does herd immunity affect disease transmission in the SIR model?",
            "What's the difference between density-dependent and frequency-dependent transmission?"
        ]
        
        for q in example_questions:
            st.markdown(f"- {q}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a footer
    st.markdown('<p class="footer-text">This assistant can help explain concepts related to the population biology models in this application.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    # This allows the file to be run directly for testing
    st.set_page_config(page_title="Population Biology Model Assistant", layout="wide")
    run()
