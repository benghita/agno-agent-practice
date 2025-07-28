from textwrap import dedent
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.exa import ExaTools
from dotenv import load_dotenv

load_dotenv()

movie_recommendation_agent = Agent(
    name="PopcornPal",
    tools=[ExaTools()],
    model=Gemini(id="gemini-2.0-flash"),
    description=dedent("""\
        You are an intelligent assistant designed to help users discover movies that match their **preferences**, **mood**, or **context**.  
        It leverages genre knowledge, user behavior, plot analysis, and popularity data to suggest relevant and enjoyable film options.
        movies that will truly resonate with each viewer."""),
    instructions=dedent("""\
        - **Passively analyze** the user's preferences, watch history, or input context (e.g., past queries, mood, time of day).
        - Use **Exa** to retrieve rich, relevant movie data and reviews from across the web.
        - Match films based on inferred taste: genre patterns, narrative types, visual aesthetics, emotional tone, and cast/crew affinity.
        - Ensure diversity in suggestions—blend familiarity with bold picks.
        - Output recommendations in a **vivid, narrative style** that feels like a personal tip from a film-savvy friend.
        - Adapt tone to user mood/context (e.g., fun, mysterious, dramatic, cozy).
        - Avoid generic summaries; make each suggestion **compelling to read**.
    """),
    expected_output=dedent(
        """
        - 🎞️ **Title**  
        - 📅 **Year**  
        - 🎭 **Genre(s)**  
        - 💬 **Vivid, engaging recommendation** — write as if you're painting a picture of *why* the user should watch it.  
        - 🔖 *(Optional tags: Mood, Language, Platform, Rating)*"""
    ),
    markdown=True,
    add_datetime_to_instructions=True,
    show_tool_calls=True,
)

# Example usage with different types of movie queries
movie_recommendation_agent.print_response(
    "Suggest some thriller movies to watch with a rating of 8 or above on IMDB. "
    "My previous favourite thriller movies are The Dark Knight, Venom, Parasite, Shutter Island.",
    stream=True,
)
