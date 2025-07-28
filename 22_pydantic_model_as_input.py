from typing import List
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.hackernews import HackerNewsTools
from pydantic import BaseModel, Field
from textwrap import dedent
from dotenv import load_dotenv

load_dotenv()

class ResearchTopic(BaseModel):
    """Structured research topic with specific requirements"""

    topic: str
    focus_areas: List[str] = Field(description="Specific areas to focus on")
    target_audience: str = Field(description="Who this research is for")
    sources_required: int = Field(description="Number of sources needed", default=5)

# Define agents
hackernews_agent = Agent(
    name="Hackernews Agent",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[HackerNewsTools()],
    role="Extract key insights and content from Hackernews posts",
    instructions=dedent(
        """
        - Scan Hacker News front page, new posts, and high-ranking threads.
        - Identify:
          - Posts gaining fast traction (based on score, velocity, and engagement)
          - Top-voted and most insightful comments
          - Recurring topics or community sentiment shifts
        - Summarize posts and threads in an engaging and **informative tone**, highlighting why each is worth attention.
        - Include links for easy exploration."""
    )
)

hackernews_agent.print_response(
    message=ResearchTopic(
        topic="AI",
        focus_areas=["AI", "LLM"],
        target_audience="Developers",
        sources_required=5,
    )
)