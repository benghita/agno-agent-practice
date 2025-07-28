from agno.agent import Agent
from agno.models.google import Gemini
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from dotenv import load_dotenv
from pprint import pprint

load_dotenv() 

memory = Memory(
    model=Gemini(id="gemini-2.0-flash"),
    db=SqliteMemoryDb(
        table_name="agent_memories",
        db_file="tmp/agent.db"
    )
)

# Get user ID from terminal
user_id = input("Enter your user ID: ")
memories = memory.get_user_memories(user_id=user_id)

agent = Agent(
    user_id=user_id,
    model=Gemini(
        id="gemini-2.0-flash"
        ),
    instructions=[
        "You are a simple AI agent designed to receive user input and respond with helpful, relevant answers.",
        "Instructions:",
        "- Understand the input and identify the intent.",
        "- Respond with a clear and concise answer.",
    ],
    # Store memories in a database
    memory=memory,
    # Give the Agent the ability to update memories
    enable_agentic_memory=True,
    markdown=True
)

memory.clear()

if __name__ == "__main__":
    
    print(f"Welcome! You are logged in as: {user_id}")
    print("Type 'quit' to exit the conversation.")
    print("-" * 50)
    
    # Interactive conversation loop
    while True:
        user_input = input(f"{user_id}: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if user_input.strip():  # Only process non-empty input
            agent.print_response(user_input)
            
            # Optionally show memories after each interaction
            pprint(memories)
            
            print(f"\nMemories about {user_id}:")
            pprint(memories)
            print("-" * 50)