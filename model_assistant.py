# model_assistant.py - Claude-powered assistant for population biology models
import streamlit as st
import anthropic
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
def get_claude_response(messages):
    """Get a response from Claude API using the API key from secrets.toml"""
    try:
        # Access the API key from secrets.toml
        api_key = st.secrets["anthropic"]["api_key"]
        
        # Create a new client with the API key
        client = anthropic.Anthropic(api_key=api_key)
        
        # System prompt for guiding Claude's responses
        system_prompt = """You are a specialized assistant for population biology and epidemiological modeling.
                
You help students and researchers understand concepts related to:
- Reed-Frost models
- Leslie Matrix models
- Macdonald models
- SIR/SEIR models
- Vector-borne disease dynamics
- Population biology principles

Provide educational, accurate responses with relevant equations, explanations, and references. 
When explaining mathematical concepts, be thorough but clear."""
        
        # Prepare messages for API call with correct format
        api_messages = []
        
        # Add the conversation history (up to last 10 messages to stay within context limits)
        for msg in messages[-10:]:
            if msg["role"] != "system":  # Skip any system messages in history
                # Format each message with content as an array of objects
                api_messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}]
                })
        
        # Make the API call with system as a top-level parameter
        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=2000,
            system=system_prompt,  # System prompt as top-level parameter
            messages=api_messages
        )
        
        # The response format has changed too, extract text content
        return response.content[0].text
    except KeyError:
        return "Error: API key not found in secrets.toml. Please check your .streamlit/secrets.toml file."
    except Exception as e:
        return f"Error connecting to Claude API: {str(e)}\n\nPlease check your API key and connection."

def run():
    """Main function to run the assistant interface"""
    
    # Apply styling
    local_css()
    
    # Initialize session state values for messages if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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
            Powered by Claude
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Configure sidebar
    with st.sidebar:
        st.subheader("Assistant Settings")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input with chat_input (prevents looping)
    user_question = st.chat_input("Ask about population biology models...")
    
    # Process new user input
    if user_question:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Display assistant response with spinner
        with st.chat_message("assistant"):
            with st.spinner("Researching population biology concepts..."):
                # Get response from Claude API
                response = get_claude_response(st.session_state.messages)
                st.markdown(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    run()
