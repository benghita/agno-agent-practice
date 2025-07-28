from agno.agent import Agent
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.embedder.google import GeminiEmbedder
from agno.models.google import Gemini
from agno.vectordb.lancedb import LanceDb
from dotenv import load_dotenv
from textwrap import dedent

load_dotenv()

knowledge = PDFUrlKnowledgeBase(
    urls=[
        "https://core.ac.uk/download/pdf/236051473.pdf",
    ],
    vector_db=LanceDb(
        table_name="legal_docs", 
        uri="tmp/lancedb",
        embedder=GeminiEmbedder(),
    )
)
knowledge.load(recreate=False)

legal_agent = Agent(
    name="LegalAdvisor",
    knowledge=knowledge,
    search_knowledge=True,
    model=Gemini(id="gemini-2.0-flash"),
    markdown=True,
    instructions=dedent(
        """
        - Use the internal **legal knowledge base** to generate responses.
        - Provide **accurate, concise, and jurisdiction-aware** legal guidance.
        - Always include **relevant references** from laws, regulations, or case examples.
        
        ## Response Structure:
        
        1. **Issue Summary**  
           - Briefly restate the user's question in legal terms.
        
        2. **Applicable Law/Policy**  
           - Identify relevant legal principles, statutes, or articles.
        
        3. **Advice & Reasoning**  
           - Provide a clear recommendation or explanation.
           - Justify using facts from the knowledge base."""
    ),
)

legal_agent.print_response(
    "What are the legal consequences and criminal penalties for illegal access to a computer?",
    stream=True,
)