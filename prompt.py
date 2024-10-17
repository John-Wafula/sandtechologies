import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
import os
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)

# Initialize the chatbot model
model = ChatGroq(model_name="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

# Define the system message template
system = '''You are a helpful assistant that answers questions based on provided context.
Your responses must be plain text, avoiding special characters like new lines, italics, tabs, or block quotes.'''

# Load the CSV and group the content by conversation
file_path = 'C:/Users/user/Documents/Sand Technology/messages_10000.csv'  # Replace with your file path
data = pd.read_csv(file_path)
grouped_content = data.groupby('conversation_id')['content'].apply(' '.join).reset_index()

def llm_query(user_input: str, context: str) -> str:
    """
    Query the model with user input and relevant context.
    """
    try:
        # Combine system message and user input with context
        human_message = f"user input: {user_input}\ndocument context: {context}"
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessagePromptTemplate.from_template(human_message),
        ])

        # Format the messages with context and user input
        messages = chat_template.format_messages(
            context=context,
            user_input=user_input
        )

        # Get the response from the model
        data = model.invoke(messages)
        for item in data:
            if item[0] == "content":
                response = item[1]
                break
        return response

    except Exception as e:
        logger.error(f"Error in querying model: {e}")
        return "An error occurred. Please try again."

def llm_answer(user_input: str) -> str:
    """
    Search the content for the most relevant context based on user input.
    """
    try:
        # Find the most relevant conversation (basic search)
        context_results = grouped_content[grouped_content['content'].str.contains(user_input, case=False, na=False)]
        context = context_results['content'].iloc[0] if not context_results.empty else "No relevant context found"

        # Get the response using the retrieved context
        return llm_query(user_input=user_input, context=context)

    except Exception as e:
        logger.error(f"Error in retrieving context: {e}")
        return "An error occurred while retrieving the context. Please try again."

# Example query
if __name__ == "__main__":
    query = "How can I add the IA calendar to my Google Calendar?"  # Replace with your query
    response = llm_answer(query)
    print("Response:", response)
