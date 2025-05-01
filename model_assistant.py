# model_assistant.py - Claude-powered assistant for population biology models
import streamlit as st
import anthropic
import os
from datetime import datetime
from random import choice

# Note: Do NOT call st.set_page_config() here since app.py already has it

def local_css():
    """Define custom CSS styles"""
    st.markdown("""
    <style>
    /* Main container styles */
    .main-container {
        background-color: rgba(17, 17, 40, 0.4);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(100, 149, 237, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Header styling */
    .custom-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #E0E0FF;
        font-weight: 600;
        font-size: 1.8rem;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Description text */
    .description-text {
        color: #B8B8FF;
        font-size: 0.95rem;
        font-weight: 300;
        line-height: 1.5;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(120, 120, 255, 0.2);
    }
    
    /* Button styling */
    .stButton button {
        background-color: rgba(100, 149, 237, 0.7) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 8px 15px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        background-color: rgba(100, 149, 237, 0.9) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Example questions styling */
    .example-button {
        background-color: rgba(70, 130, 230, 0.6) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        margin-bottom: 8px !important;
        text-align: center !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        padding: 12px 20px !important;
        font-size: 0.95rem !important;
    }
    
    .example-button:hover {
        background-color: rgba(70, 130, 230, 0.8) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Chat input styling - ENHANCED & BIGGER */
    .stChatInputContainer {
        border-radius: 25px !important;
        background-color: rgba(100, 149, 237, 0.15) !important;
        border: 3px solid rgba(120, 120, 255, 0.4) !important;
        padding: 10px !important;
        margin-top: 25px !important;
        margin-bottom: 30px !important;
        box-shadow: 0 6px 15px rgba(0, 0, 100, 0.15) !important;
    }
    
    /* Make the input text larger */
    .stChatInputContainer input {
        font-size: 1.1rem !important;
        padding: 15px 20px !important;
        height: 60px !important;
    }
    
    /* Pulse animation for the chat input */
    @keyframes gentle-pulse {
        0% { box-shadow: 0 0 0 0 rgba(100, 149, 237, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(100, 149, 237, 0); }
        100% { box-shadow: 0 0 0 0 rgba(100, 149, 237, 0); }
    }
    
    /* Chat input focus styling */
    .stChatInputContainer:focus-within {
        border: 3px solid rgba(120, 120, 255, 0.8) !important;
        box-shadow: 0 6px 20px rgba(100, 149, 237, 0.35) !important;
        animation: gentle-pulse 2s infinite;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 18px !important;
        padding: 12px 18px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 50, 0.1) !important;
    }
    
    /* User message styling */
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: rgba(100, 149, 237, 0.15) !important;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: rgba(120, 120, 200, 0.1) !important;
    }
    
    /* Tip box styling */
    .tip-box {
        background-color: rgba(100, 149, 237, 0.15);
        border-left: 4px solid rgba(100, 149, 237, 0.7);
        padding: 10px 15px;
        border-radius: 0 10px 10px 0;
        margin: 15px 0;
        font-size: 0.9rem;
        color: #D0D0FF;
    }
    
    /* Chat prompt styling - making it more prominent */
    .chat-prompt-container {
        background-color: rgba(100, 149, 237, 0.15);
        border-radius: 20px;
        padding: 15px;
        margin: 30px 0 20px 0;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .chat-prompt-text {
        color: #E0E0FF;
        font-size: 1.2rem;
        font-weight: 500;
        margin: 0;
    }
    
    .chat-prompt-icon {
        font-size: 1.5rem;
        margin-right: 10px;
    }
    
    /* Sample questions container */
    .sample-container {
        background-color: rgba(100, 149, 237, 0.07);
        border-radius: 15px;
        padding: 20px;
        margin: 25px 0;
    }
    
    .sample-header {
        color: #D0D0FF;
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }
    
    .sample-header-icon {
        margin-right: 10px;
        font-size: 1.4rem;
    }
    
    /* Footer styling */
    .footer-text {
        color: #8888BB;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 20px;
        padding-top: 10px;
        border-top: 1px solid rgba(120, 120, 255, 0.1);
    }
    
    /* Hide the extra empty sections/containers */
    section.main > div:first-of-type > div:nth-child(1),
    section.main > div:first-of-type > div:nth-child(7) {
        display: none !important;
    }
    
    /* Sidebar styling */
    .css-1544g2n {
        padding-top: 2rem;
    }
    
    .sidebar-header {
        color: #E0E0FF;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

def get_random_tip():
    """Return a random tip about population models"""
    tips = [
        "The basic reproduction number (R₀) indicates how contagious an infectious disease is.",
        "Reed-Frost models are useful for studying disease spread in small populations.",
        "Leslie Matrix models help understand age-structured population dynamics.",
        "The Macdonald model is essential for studying vector-borne diseases like malaria.",
        "SIR models divide the population into Susceptible, Infected, and Recovered groups.",
        "Leslie matrices can predict future population sizes based on age-specific fertility and survival rates.",
        "Vector-borne diseases require understanding both host and vector population dynamics.",
        "Population thresholds are critical points where disease dynamics change dramatically.",
    ]
    return choice(tips)

def run():
    """Main function to run the assistant interface"""
    
    # Apply styling
    local_css()
    
    # Initialize session state for chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Configure sidebar for the assistant settings
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Assistant Settings</div>', unsafe_allow_html=True)
        
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
    
    # Function to use example question
    def use_example_question(question):
        st.session_state.example_question = question
        st.experimental_rerun()
    
    # Display chat interface with only ONE header
    # Only render the header here if there are NO messages yet
    if len(st.session_state.messages) == 0:
        # Single header with DNA icon
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown('<div class="custom-header">🧬 Population Biology Model Assistant</div>', unsafe_allow_html=True)
        st.markdown('<p class="description-text">Ask questions about population biology models, epidemiology concepts, or get help with interpreting model results. This assistant has internet access to provide up-to-date information.</p>', unsafe_allow_html=True)
        
        # Add a tip box
        st.markdown(f'<div class="tip-box">💡 <b>Did you know?</b> {get_random_tip()}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat container
    main_container = st.container()
    with main_container:
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # If no messages yet, show example questions with clickable behavior
    if len(st.session_state.messages) == 0:
        st.markdown('<div class="sample-container">', unsafe_allow_html=True)
        st.markdown('<div class="sample-header"><span class="sample-header-icon">👋</span> Try asking one of these questions:</div>', unsafe_allow_html=True)
        
        example_questions = [
            "How does a Reed-Frost model work?",
            "Explain the Leslie Matrix approach for modeling population dynamics",
            "What factors influence the basic reproduction number (R₀) in vector-borne diseases?",
            "How does herd immunity affect disease transmission in the SIR model?",
            "What's the difference between density-dependent and frequency-dependent transmission?"
        ]
        
        # Create clickable example questions
        for q in example_questions:
            if st.button(q, key=f"btn_{q}", use_container_width=True, 
                        help="Click to ask this question",
                        type="primary"): 
                # This will be handled by the chat input below
                st.session_state.example_question = q
                st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a friendly prompt above the chat input
    st.markdown('<div class="chat-prompt-container">', unsafe_allow_html=True)
    st.markdown('<span class="chat-prompt-icon">💬</span><p class="chat-prompt-text">Type your question here or click one of the examples above!</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Get example question from state if exists
    example_q = ""
    if hasattr(st.session_state, 'example_question') and st.session_state.example_question:
        example_q = st.session_state.example_question
        # Clear it to avoid reusing
        st.session_state.example_question = ""
    
    # Handle user input with enhanced visual cue - BIGGER search bar
    if prompt := st.chat_input("Ask about population models, disease dynamics, or model interpretation...", key="chat_input"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response with a spinner
        with st.chat_message("assistant"):
            with st.spinner("Researching population biology concepts..."):
                # Check if API key is provided
                if not api_key and "ANTHROPIC_API_KEY" not in st.secrets:
                    response = "Please enter your Anthropic API key in the sidebar to continue."
                else:
                    # Use API key from session or secrets
                    if not api_key and "ANTHROPIC_API_KEY" in st.secrets:
                        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
                        
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
    
    # Handle example question if one was selected
    elif example_q:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": example_q})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(example_q)
        
        # Display assistant response with a spinner
        with st.chat_message("assistant"):
            with st.spinner("Researching population biology concepts..."):
                # Check if API key is provided
                if not api_key and "ANTHROPIC_API_KEY" not in st.secrets:
                    response = "Please enter your Anthropic API key in the sidebar to continue."
                else:
                    # Use API key from session or secrets
                    if not api_key and "ANTHROPIC_API_KEY" in st.secrets:
                        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
                        
                    # Generate response
                    response = get_claude_response(
                        prompt=example_q,
                        history=st.session_state.messages[:-1],  # Exclude the current message
                        system_message=system_message,
                        model=selected_model,
                        use_web_search=use_web_search
                    )
                
                # Display response
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Add a subtle footer only if we have messages
    if len(st.session_state.messages) > 0:
        st.markdown('<p class="footer-text">This assistant can help explain concepts related to the population biology models in this application.</p>', unsafe_allow_html=True)
