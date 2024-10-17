import streamlit as st
import pandas as pd
import random
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
import os
import logging
from dotenv import load_dotenv
from langchain.evaluation.criteria import CriteriaEvalChain
from langchain_openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = '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'
os.environ['OPENAI_API_KEY'] = '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'  # Add the OpenAI key

# Initialize the chatbot model
model = ChatGroq(model_name="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

# Initialize the evaluation model (for response evaluation)
eval_model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the system message template
system = '''You are a helpful assistant that answers questions based on provided context. You help 
ALX students who have queries on ALX learning experience. Your responses must be plain text, 
avoiding special characters like new lines, italics, tabs, or block quotes. Be as human as possible.'''

# Define evaluation criteria
criteria = {
    "accuracy": "Does the response accurately address the user's question?",
    "truthfulness": "Is the information provided in the response true?",
    "context awareness": "Is the response aware of the context provided?",
    "fluency": "Is the response fluent and free of grammatical errors?",
    "coherence": "Is the response coherent and logically structured?",
    "naturalness of language": "Does the response use natural, human-like language?"
}

# Initialize the evaluation chain
evaluator = CriteriaEvalChain.from_llm(llm=eval_model, criteria=criteria)

# Function to evaluate the chatbot's response
def evaluate_response(user_input, response, context):
    """
    Evaluate the response based on predefined criteria.
    """
    try:
        evaluation = evaluator.evaluate_strings(prediction=response, input=f"{user_input}\n{context}")
        return evaluation
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return {"error": "An error occurred during evaluation."}

# Load the CSV file and group content by conversation
csv_file_path = 'C:/Users/user/Documents/Sand Technology/messages_10000.csv'
df = pd.read_csv(csv_file_path)
grouped_content = df.groupby('conversation_id')['content'].apply(' '.join).reset_index()

# Initialize a dictionary to store conversations
conversations = {}

# Populate the conversations dictionary
for _, row in df.iterrows():
    conversation_id = row['conversation_id']
    content = row['content']
    ordinality = row['ordinality']
    
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    conversations[conversation_id].append((ordinality, content))

# Function to fetch and display a conversation by ID
def fetch_and_display_conversation(conversation_id):
    if conversation_id not in conversations:
        st.warning("Conversation ID not found. Please try again.")
        return

    messages = sorted(conversations[conversation_id])
    for i, (ordinality, content) in enumerate(messages):
        sender = "Bot" if i % 2 == 0 else "Student"
        st.write(f"**{sender}**: {content}")
    st.write("\n---\n")

# Function to add rating and mark review section
def rate_and_mark_review(conversation_id):
    st.subheader("Rate this Conversation")
    rating = st.slider(f"Rate the conversation {conversation_id}", 1, 5, value=3, key=f"rating_{conversation_id}")
    
    if st.button(f"Mark Conversation {conversation_id} as Reviewed", key=f"review_button_{conversation_id}"):
        st.session_state[f"reviewed_{conversation_id}"] = True
        st.success(f"Conversation {conversation_id} has been marked as reviewed!")

# Function to query the model with user input and context
def llm_query(user_input, context):
    try:
        human_message = f"user input: {user_input}\ndocument context: {context}"
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessagePromptTemplate.from_template(human_message),
        ])
        messages = chat_template.format_messages(context=context, user_input=user_input)
        data = model.invoke(messages)
        for item in data:
            if item[0] == "content":
                response = item[1]
                break
        return response

    except Exception as e:
        logger.error(f"Error in querying model: {e}")
        return "An error occurred. Please try again."

# Function to get the response using the retrieved context
def llm_answer(user_input):
    try:
        context_results = grouped_content[grouped_content['content'].str.contains(user_input, case=False, na=False)]
        context = context_results['content'].iloc[0] if not context_results.empty else "No relevant context found"
        return llm_query(user_input=user_input, context=context), context

    except Exception as e:
        logger.error(f"Error in retrieving context: {e}")
        return "An error occurred while retrieving the context. Please try again.", ""

# Streamlit UI
st.title("ALX Conversation Viewer and Chatbot")

# Sidebar for user input
st.sidebar.title("Options")
st.sidebar.header("Instructions")
st.sidebar.write("""
- To view a specific conversation, enter the Conversation ID in the field below and click 'Fetch Conversation'.
- To review a random unreviewed conversation, click 'Show Random Conversation'.
- You can rate the conversation and mark it as reviewed after viewing it.
""")

# Input field to enter a conversation ID in the sidebar
conversation_id_input = st.sidebar.text_input("Enter Conversation ID")

# Button to fetch the conversation
if st.sidebar.button("Fetch Conversation"):
    st.session_state['current_conversation_id'] = conversation_id_input

# Retrieve and display the stored conversation ID if it exists in session state
if 'current_conversation_id' in st.session_state:
    current_conversation_id = st.session_state['current_conversation_id']
    st.write(f"### Conversation: {current_conversation_id}")
    fetch_and_display_conversation(current_conversation_id)
    rate_and_mark_review(current_conversation_id)

# Optional: Button to show a random conversation
if st.sidebar.button("Show Random Conversation"):
    random_id = random.choice(list(conversations.keys()))
    st.session_state['current_conversation_id'] = random_id
    st.write(f"### Randomly Selected Conversation ID: {random_id}")
    fetch_and_display_conversation(random_id)
    rate_and_mark_review(random_id)

# Optional: Button to clear session state (for testing purposes)
if st.sidebar.button("Reset Reviewed State"):
    st.session_state.clear()
    st.success("Session state has been reset. You can now mark conversations as reviewed again.")

# Chatbot interface
st.header("Chat with the Bot")

# Input field for user query
user_query = st.text_input("Ask a question:")

# Initialize variables to hold response and context
if 'bot_response' not in st.session_state:
    st.session_state.bot_response = ""
if 'response_context' not in st.session_state:
    st.session_state.response_context = ""

# Button to get the response from the chatbot
if st.button("Send"):
    if user_query:
        st.session_state.bot_response, st.session_state.response_context = llm_answer(user_query)
        st.write("**Bot**: " + st.session_state.bot_response)
    else:
        st.warning("Please enter a question to ask the bot.")

# Add a flag to check if the response is generated
response_generated = bool(st.session_state.bot_response)

# Evaluation button (only show if a response has been generated)
if response_generated:
    if st.button(f"Evaluate Response for this response"):
        if st.session_state.bot_response and st.session_state.bot_response != "An error occurred. Please try again.":
            evaluation = evaluate_response(user_query, st.session_state.bot_response, st.session_state.response_context)
            
            # Formatting the evaluation results
            reasoning = evaluation.get("reasoning", "")
            value = evaluation.get("value", "")
            score = evaluation.get("score", 0)

            # Create a structured string for the evaluation results
            evaluation_summary = f"""
            
            **Evaluation Results:**

             {reasoning}

             {value}
            

            **Score:** {score}
               

            
            """

            st.markdown(evaluation_summary)  # Use markdown for better formatting
        else:
            st.warning("No valid response to evaluate.")
