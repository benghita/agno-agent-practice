from agno.agent import Agent
from agno.models.google import Gemini
from agno.vectordb.lancedb import LanceDb
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.wikipedia import WikipediaKnowledgeBase
from dotenv import load_dotenv

load_dotenv() 

# Creating knowladge from Wikipedia pages 
knowledge_base = WikipediaKnowledgeBase(
    topics=["Artificial Intelligence", "Large Language Model"],
   
    # Table name: wikipedia_documents
    vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="wikipedia_documents",
            embedder=GeminiEmbedder(),
        ),
)

wiki_agent = Agent(
    model=Gemini(
        id="gemini-2.0-flash"
        ),
    instructions=[
        "You assist users by answering questions using a knowledge base of AI-related Wikipedia pages stored in a vector database.",
        "- Search the vector database for the most relevant information.",
        "- Summarize or extract a clear answer from the retrieved content."
    ],
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True
)

if __name__ == "__main__":
    wiki_agent.knowledge.load(recreate=False)
    wiki_agent.print_response(
        "What is the history of neural networks?",
        stream=True,
        show_full_reasoning=True,
        stream_intermediate_steps=True,
    )