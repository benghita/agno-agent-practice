from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv

load_dotenv() 

basic_agent = Agent(
    model=Gemini(
        id = "gemini-2.0-flash",
    ),
    instructions=[
        "You are a simple AI agent designed to receive user input and respond with helpful, relevant answers.",
        "Instructions:",
        "- Understand the input and identify the intent.",
        "- Respond with a clear and concise answer.",
    ],
    show_tool_calls=True,
)

if __name__ == "__main__":
    basic_agent.print_response(
        "How do I convert Celsius to Fahrenheit?",
        stream=True,
        show_full_reasoning=True,
        stream_intermediate_steps=True,
    )