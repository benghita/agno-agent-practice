from agno.agent import Agent
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.url import UrlKnowledge
from agno.models.google import Gemini
from agno.storage.sqlite import SqliteStorage
from agno.vectordb.lancedb import LanceDb, SearchType
from dotenv import load_dotenv

load_dotenv()

knowledge = UrlKnowledge(
    urls=["https://docs.agno.com/applications/fastapi/introduction.md"],
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="agno_assist",
        search_type=SearchType.hybrid,
        embedder=GeminiEmbedder(),
    ),
)

storage = SqliteStorage(
    table_name="agno_assist_sessions", 
    db_file="tmp/agent.db"
)

agno_assist = Agent(
    name="Agno Assist",
    model=Gemini(id="gemini-2.0-flash"),
    description="You help answer questions about the Agno framework.",
    instructions="Search your knowledge before answering the question.",
    knowledge=knowledge,
    storage=storage,
    add_history_to_messages=True,
    add_datetime_to_instructions=True,
    markdown=True,
)

if __name__ == "__main__":
    agno_assist.knowledge.load()  # Load the knowledge base, comment after first run
    agno_assist.print_response("How to host agents as FastAPI Applications?")