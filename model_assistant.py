# model_assistant.py - Claude-powered assistant for population biology models
import streamlit as st
import anthropic
import os
from datetime import datetime

def run():
    """Main function to run the assistant interface"""
    
    # Initialize session state for chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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
    
    # Add a footer
    st.caption("This assistant can help explain concepts related to the population biology models in this application.")

if __name__ == "__main__":
    # This allows the file to be run directly for testing
    st.set_page_config(page_title="Population Biology Model Assistant", layout="wide")
    run()
