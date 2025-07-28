from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.exa import ExaTools
from dotenv import load_dotenv
from textwrap import dedent

load_dotenv()

agent = Agent(
    name="shopping partner",
    model=Gemini(id="gemini-2.0-flash"),
    description=dedent(
        """
        You are a smart shopping assistant that helps users **find the best products and deals** based on what they want to buy.  
        When a user tells you what they’re looking for, you search the web using tools like **Exa** to find **reliable, up-to-date, and well-priced** options.  
        Your goal is to **save time and money**, offering high-quality recommendations with purchase links and relevant comparisons."""
    ),
    instructions=[
        "- Accept input as a shopping need or product description (e.g., 'I need a lightweight laptop under $800" or "Looking for a non-stick frying pan').",
        "- Use **Exa** or similar tools to search product listings, online stores, and reviews.",
        "- Return **top 2–3 options** that match the user's preferences in:",
        "  - Price range",
        "  - Brand reliability",
        "  - Feature quality",
        "  - User reviews",
        "- Provide clear descriptions and **direct links** to the offers.",
        "- Include pros/cons, availability, and price.",
        "- Avoid recommending outdated or unavailable items.",
        "- Write in a helpful, friendly tone that inspires confidence.",
    ],
    tools=[ExaTools()],
    show_tool_calls=True,
)
agent.print_response(
    "I need a good pair of wireless noise-cancelling headphones under $200"
)