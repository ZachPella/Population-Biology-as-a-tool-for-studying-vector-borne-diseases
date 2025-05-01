# model_assistant.py - Claude-powered assistant for population biology models
import streamlit as st
import anthropic
import os
from datetime import datetime

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
        font-size: 1.6rem;
        margin: 30px 0;
        text-align: center;
        font-weight: 500;
    }
    
    /* Question label */
    .question-label {
        font-size: 3rem;
        margin-bottom: 10px;
        color: #333;
    }
    
    /* Example questions styling */
    .example-button {
        width: 100%;
        text-align: left;
        background-color: #f0f2f6;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 10px 15px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .example-button:hover {
        background-color: #e0e2e6;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 60px;
        text-align: center;
        color: #555;
        font-size: 1.5rem;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .linkedin-icon {
        margin-left: 10px;
    }
    
    /* Streamlit elements customization */
    div[data-testid="stTextInput"] input {
        height: 50px;
        font-size: 1.1rem;
        padding: 10px 15px;
        border-radius: 4px;
    }
    
    .stChatMessage {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    /* Chat message styling */
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
    
    /* Button styling */
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

def run():
    """Main function to run the assistant interface"""
    
    # Apply styling
    local_css()
    
    # Initialize session state for chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Title with robot emoji
    st.markdown("""
    <div class="title-container">
        <h1 class="main-title">Population Biology Model Assistant</h1>
        <span class="robot-icon">🤖</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Subtitle
    st.markdown('<p class="subtitle">Ask questions about population biology models:</p>', unsafe_allow_html=True)
    
    # Configure sidebar for the assistant settings
    with st.sidebar:
        st.subheader("Assistant Settings")
        
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
    
    # Main container for question input
    st.markdown('<p class="question-label">Your Question:</p>', unsafe_allow_html=True)
    
    # User input field - larger and more prominent
    user_question = st.text_input("", 
                                label_visibility="collapsed", 
                                placeholder="Type your question about population biology models...",
                                key="user_question_input")
    
    # If user has entered a question
    if user_question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Reset the input field
        st.session_state.user_question_input = ""
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
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
                        prompt=user_question,
                        history=st.session_state.messages[:-1],  # Exclude the current message
                        system_message=system_message,
                        model=selected_model,
                        use_web_search=use_web_search
                    )
                
                # Display response
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
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
