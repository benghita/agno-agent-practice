from agno.agent import Agent
from agno.models.google import Gemini
from agno.storage.sqlite import SqliteStorage
from dotenv import load_dotenv

load_dotenv() 

# Creating the storage for the session
storage = SqliteStorage(
    table_name="sessions",
    db_file="tmp/agent.db"
)

agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    instructions=[
        "You are a simple AI agent designed to receive user input and respond with helpful, relevant answers.",
        "Instructions:",
        "- Understand the input and identify the intent.",
        "- Respond with a clear and concise answer.",
    ],
    show_tool_calls=True,
    storage=storage,
    # Add the chat history to the messages
    add_history_to_messages=True,
    # Number of history runs
    num_history_runs=3,
    markdown=True
)

if __name__ == "__main__":
    agent.print_response("I’m building an AI chatbot.")
    agent.print_response("What framework should I use?")
