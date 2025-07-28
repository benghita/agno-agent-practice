import asyncio
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.mcp import MCPTools
from agno.tools.thinking import ThinkingTools
from dotenv import load_dotenv

load_dotenv()

async def run_agent(message: str) -> None:
    async with MCPTools(
        "npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt",
        timeout_seconds=20
    ) as mcp_tools:
        agent = Agent(
            description="You are an intelligent assistant connected to the Airbnb API. You help users find and book short-term rentals based on their preferences.",
            model=Gemini(id="gemini-2.0-flash"),
            tools=[ThinkingTools(), mcp_tools],
            instructions=[
                "1. Analyse user input.",
                "2. Query the Airbnb API for matching listings.",
                "3. Show top results with price, rating, and key features.",
                "4. Support follow-up filters or booking actions.",
                "5. Always provide links and update availability.",
            ],
            add_datetime_to_instructions=True,
            show_tool_calls=True,
            markdown=True,
        )
        await agent.aprint_response(message, stream=True)


if __name__ == "__main__":
    task = "Any entire homes in Marrakech with a kitchen and great reviews for the next 3 days?"
    asyncio.run(run_agent(task))