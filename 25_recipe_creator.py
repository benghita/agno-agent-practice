from textwrap import dedent
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv

load_dotenv()

recipe_generator_agent = Agent(
    name="ChefGenius",
    tools=[DuckDuckGoTools()],
    model=Gemini(id="gemini-2.0-flash"),
    description=dedent("""\
        You are an intelligent assistant that generates personalized recipes based on the user's **available ingredients** and optionally their **time constraints**.  
        Your goal is to help users make the most of what they have at home by creating **delicious, easy-to-follow recipes** with clear cooking instructions and timing.
        """),
    instructions=dedent("""\
        - Analyze the user’s input:
          - List of available ingredients (e.g., “eggs, spinach, tomatoes, bread”)
          - Optional: maximum cooking time (e.g., “under 20 minutes”)
        
        - Use duckduckgo to search for any famous or special recepes.
        - Match ingredients to suitable meals using a recipe database or your knowledge.
        - Ensure the recipe uses **mostly or only the provided ingredients**, unless adding 1–2 common staples (like salt, oil, butter) for realism.
        - Adapt the cooking method to fit the time constraint.
        - Output a recipe that includes:
          - Dish name
          - Cooking time
          - Ingredients list (quantified if possible)
          - Step-by-step cooking instructions
          - Serving size
          - Optional tips or variation ideas
        
        - Write in an encouraging and easy-to-understand tone.
        """),
    markdown=True,
    add_datetime_to_instructions=True,
    show_tool_calls=True,
)

# Example usage with different types of recipe queries
recipe_generator_agent.print_response(
    "I have onion, soya souce, garlic, egg and rice. Give an easy recipe, with no gluten.",
    stream=True,
)