from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
load_dotenv() 

agent = Agent(
    model=Gemini(),
    markdown=True,
)

agent.print_response('Explain CRISPR in simple terms')