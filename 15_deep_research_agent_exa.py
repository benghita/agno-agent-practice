from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.exa import ExaTools
from dotenv import load_dotenv
from textwrap import dedent

load_dotenv()

agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    tools=[ExaTools(research=True)],
    instructions=dedent("""
        You are a structured research assistant focused on extracting factual information from Wikipedia and writing detailed articles.

        You can use the Wikipedia tool for structured lookups.

        The search tool accepts:
        - `query` (str): The search term or topic
        - `output_schema` (dict, optional): A JSON schema defining how the output should be structured.

        If the user gives you a schema, pass it along with the query.

        If no schema is provided, just pass the topic and return a brief summary.

        Format the result clearly and concisely.
    """),
    show_tool_calls=True,
    markdown=True
)

# Example call
#agent.print_response(
#    "Write a detailed article about the history and evolution of the Internet. Include key milestones, important figures, and major technological changes."
#)

# Define a JSON schema for structured research output
research_schema = {
    "type": "object",
    "properties": {
        "major_players": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "contributions": {"type": "string"},
                },
            },
        },
    },
    "required": ["major_players"],
}
agent.print_response(
    f"Research the top 3 Semiconductor companies in 2024. Use this schema {research_schema}."
)
