from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv

load_dotenv() 

search_assistant = Agent(
    model=Gemini(
        id = "gemini-2.0-flash",
    ),
    instructions=[
        "You are a helpful assistant designed to search the web and provide relevant information based on the user's queries.",
        "Instructions:",
        "- Search the web for accurate and recent information",
        "- Summarize the key findings clearly and concisely",
        "- Provide sources or links if available."
    ],
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
)


if __name__ == "__main__":
    search_assistant.print_response(
        "List the main types of machine learning models.",
        stream=True,
        show_full_reasoning=True,
        stream_intermediate_steps=True,
    )