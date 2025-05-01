# Internet-Connected Claude Chatbot for Streamlit
# This application integrates a Claude-powered chatbot with web search capabilities
# into your existing Streamlit app to help students answer questions.

import streamlit as st
import anthropic
import os
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Classroom Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sidebar_expanded" not in st.session_state:
    st.session_state.sidebar_expanded = False

# Function to toggle sidebar
def toggle_sidebar():
    st.session_state.sidebar_expanded = not st.session_state.sidebar_expanded

# Configure sidebar
with st.sidebar:
    st.title("Classroom Assistant Settings")
    
    # API key input
    api_key = st.text_input("Anthropic API Key", type="password")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    
    # Model selection
    model_options = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
    selected_model = st.selectbox("Choose Claude Model", model_options)
    
    # Temperature adjustment
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # System message customization
    system_message = st.text_area(
        "System Message", 
        value="You are a helpful classroom assistant that helps students answer questions. You're connected to the internet so you can find up-to-date information. Always provide educational, accurate responses appropriate for students.",
        height=150
    )
    
    # Web search toggle
    use_web_search = st.toggle("Enable Web Search", value=True)
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main content area
st.title("Classroom Assistant")
st.caption("I can help answer questions using the latest information from the internet!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to generate response from Claude
def get_claude_response(prompt, history, system_message, model, temperature, use_web_search):
    try:
        client = anthropic.Client(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        # Prepare messages for Claude
        messages = [{"role": "system", "content": system_message}]
        
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
                    "description": "Search the web for information",
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
            max_tokens=1024,
            temperature=temperature,
            system=system_message,
            messages=messages[1:],  # Skip system message as it's passed separately
            tools=tools
        )
        
        return response.content[0].text
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Handle user input
if prompt := st.chat_input("Ask your question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
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
                    temperature=temperature,
                    use_web_search=use_web_search
                )
            
            # Display response
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a footer
st.markdown("---")
st.caption(f"Classroom Assistant powered by Claude • {datetime.now().strftime('%Y-%m-%d')}")
