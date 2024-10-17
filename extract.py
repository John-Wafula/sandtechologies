import pandas as pd

# Load the data from a CSV file
df = pd.read_csv('C:/Users/user/Documents/Sand Technology/messages_10000.csv')

# Initialize a dictionary to store conversations
conversations = {}

# Iterate over the rows in the dataframe
for _, row in df.iterrows():
    # Extract conversation_id, content, and ordinality
    conversation_id = row['conversation_id']
    content = row['content']
    ordinality = row['ordinality']
    
    # Create a new conversation if it doesn't exist
    if conversation_id not in conversations:
        conversations[conversation_id] = []

    # Append the message to the corresponding conversation
    conversations[conversation_id].append((ordinality, content))

# Output conversations
for conversation_id, messages in conversations.items():
    print(f"### Conversation: {conversation_id}")
    for i, (ordinality, content) in enumerate(sorted(messages)):
        sender = "Bot" if i % 2 == 0 else "Student"  # Alternates between Bot and Student
        print(f"{sender}: {content}")
    print("\n---\n")
