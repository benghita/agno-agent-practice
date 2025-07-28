from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.reasoning import ReasoningTools
from dotenv import load_dotenv

load_dotenv() 

post_creator = Agent(
    model=Gemini(
        id="gemini-2.0-flash",
    ),
    description="You create short, engaging social media posts based on facts or topics provided by the user.",
    instructions=[
        "Search the web for recent facts, stats, or news related to the topic.",
        "Create a short, engaging post (1–3 sentences) suitable for platforms like LinkedIn or X.",
        "Make the tone informative and attention-grabbing."
    ],
    tools=[
        DuckDuckGoTools(),
        ReasoningTools(add_instructions=True),
    ],
    show_tool_calls=True,
    markdown=True
)

if __name__ == "__main__":
    topic = input("Choise a topic to create a social media post:")
    post_creator.print_response(
        topic,
        show_full_reasoning=True,
        stream=True
    )