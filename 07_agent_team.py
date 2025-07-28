from agno.agent import Agent
from agno.team.team import Team
from agno.models.google import Gemini
from agno.tools.wikipedia import WikipediaTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.reasoning import ReasoningTools
from dotenv import load_dotenv

load_dotenv() 

wiki_agent = Agent(
    name="Wikipedia Science Searcher",
    role="Find and summarize scientific information from Wikipedia.",
    model=Gemini(
        id="gemini-2.0-flash",
    ),
    tools=[
        WikipediaTools()
    ],
    show_tool_calls=True,
    markdown=True
)

web_agent = Agent(
    name="Scientific Article Searcher",
    role="Search the web for recent scientific articles, papers, and news based on the user’s",
    model=Gemini(
        id="gemini-2.0-flash",
    ),
    tools=[
        DuckDuckGoTools(),
        ReasoningTools(add_instructions=True),
    ],
    show_tool_calls=True,
    markdown=True
)

leader = Team(
    members=[wiki_agent, web_agent],
    model=Gemini(
        id="gemini-2.0-flash",
    ),
    instructions=[
        "Send the query to The Wikipedia Agent for background and basic concepts and to The Scientific Article Agent for up-to-date research or news.",
        "Combine the results into a structured answer with two sections, Foundational Knowledge,  Recent Research.",
        "Summarize in a clear, scientific tone appropriate for curious readers or students."
    ],
    markdown=True
)

if __name__ == "__main__":
    leader.print_response("What is photosynthesis?")