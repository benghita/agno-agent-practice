from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
from textwrap import dedent

load_dotenv()

lib_agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    description="Suggest books based on the user's interests, mood, or preferences. Tailor recommendations to genres, past favorites, or reading goals.",
    instructions=dedent(
        """1. Analyze the user's input, such as preferred genres, favorite authors/books, or reading goals (e.g., "learn more about philosophy" or "looking for something light and funny").
            2. Search book databases or APIs like Google Books or Goodreads to find the most relevant and well-rated matches.
            3. Recommend 3–5 books in a **table format**, including:
               - Title  
               - Author  
               - Genre  
               - Short Description or Summary """
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)

if __name__ == "__main__":
    lib_agent.print_response("I'm looking for some inspiring non-fiction books, something like *Atomic Habits* or *Deep Work*.")