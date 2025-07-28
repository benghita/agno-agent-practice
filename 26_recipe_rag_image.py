from pathlib import Path
from agno.agent import Agent
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.google import Gemini
from agno.tools.openai import OpenAITools
from agno.utils.media import download_image
from agno.vectordb.lancedb import LanceDb
from dotenv import load_dotenv

load_dotenv()

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="embed_vision_documents",
        embedder=GeminiEmbedder(),
    ),
)
knowledge_base.load()

agent = Agent(
    name="EmbedVisionRAGAgent",
    model=Gemini(),
    tools=[OpenAITools()],
    knowledge=knowledge_base,
    instructions=[
        "You are a specialized recipe assistant.",
        "When asked for a recipe:",
        "1. Search the knowledge base to retrieve the relevant recipe details.",
        "2. Analyze the retrieved recipe steps carefully.",
        "3. Use the `generate_image` tool to create a visual, step-by-step image manual for the recipe.",
        "4. Present the recipe text clearly and mention that you have generated an accompanying image manual. Add instructions while generating the image.",
    ],
    markdown=True,
    debug_mode=True,
)

agent.print_response(
    "What is the recipe for a Thai curry?",
)

response = agent.run_response
if response.images:
    download_image(response.images[0].url, Path("tmp/recipe_image.png"))