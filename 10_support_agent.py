# ===================== Imports =====================
# Standard library imports
import asyncio  # For running asynchronous code
import json     # For handling JSON data
import os       # For accessing environment variables
from textwrap import dedent  # For formatting multi-line strings

# agno-agent framework imports
from agno.agent import Agent  # Main Agent class
from agno.models.google import Gemini  # Google Gemini LLM model
from agno.embedder.google import GeminiEmbedder  # Embedding model for vector DB
from agno.storage.sqlite import SqliteStorage  # SQLite-based storage for agent sessions
from agno.tools.reasoning import ReasoningTools  # Reasoning tools for the agent
from agno.vectordb.lancedb import LanceDb  # Vector DB for knowledge base
from agno.tools.mcp import MCPTools  # Notion MCP tools for Notion integration
from mcp import StdioServerParameters  # Parameters for MCP server connection
from agno.tools.wikipedia import WikipediaTools, WikipediaKnowledgeBase  # Wikipedia tools and KB
from agno.tools.arxiv import ArxivTools  # Arxiv research tool
from agno.tools.pubmed import PubmedTools  # Pubmed research tool
from dotenv import load_dotenv  # For loading environment variables from .env file

# ===================== Load Environment Variables =====================
load_dotenv()  # Loads variables from a .env file into environment

async def run_agent():
    """
    Initializes and runs the Research Assistant Agent.
    This agent is equipped with a suite of tools for conducting research
    and saving the results to a Notion workspace via an MCP.
    """
    # --- API Key Configuration ---
    # It's crucial to set these environment variables before running the script.
    notion_token = os.getenv("NOTION_API_KEY")  # Notion API key from environment

    if not notion_token:
        # Raise an error if the Notion API key is missing
        raise ValueError(
            "Missing Notion API key. Please set the NOTION_API_KEY environment variable."
        )

    # --- Notion MCP Server Setup ---
    # This configures the connection to the Notion MCP server, which allows
    # the agent to interact with your Notion pages and databases.
    command = "npx"  # Node.js package runner
    args = ["-y", "@notionhq/notion-mcp-server"]  # Arguments to start the Notion MCP server
    env = {
        # Set required headers for Notion API authentication
        "OPENAPI_MCP_HEADERS": json.dumps(
            {"Authorization": f"Bearer {notion_token}", "Notion-Version": "2022-06-28"}
        )
    }
    #server_params = StdioServerParameters(command=command, args=args, env=env)

    # --- Agent Storage Setup ---
    # Configure persistent storage for agent sessions using SQLite
    storage = SqliteStorage(
        table_name="sessions",
        db_file="tmp/agent.db"
    )

    # --- Knowledge Base Setup ---
    # Preload Wikipedia knowledge base on key topics, backed by a vector DB
    knowledge_base = WikipediaKnowledgeBase(
        topics=["Artificial Intelligence", "Large Language Model"],
        # Table name: wikipedia_documents
        vector_db=LanceDb(
                uri="tmp/lancedb",
                table_name="wikipedia_documents",
                embedder=GeminiEmbedder(),
            ),
    )

    # --- Notion MCP Server Parameters ---
    server_params = StdioServerParameters(command=command, args=args, env=env)

    # --- Tool Initialization ---
    # Initialize all the tools the agent will have access to.
    # This includes the Notion MCP tools and all the research tools.
    async with MCPTools(server_params=server_params, timeout_seconds=20) as mcp_tools:

        # Combine all tools into a single list for the agent
        all_tools = [
            mcp_tools,           # Notion integration tools
            WikipediaTools(),    # Wikipedia search tool
            ArxivTools(),        # Arxiv research tool
            PubmedTools(),       # Pubmed research tool
            ReasoningTools()     # General reasoning tools
        ]

        # --- Agent Definition ---
        # Define the agent's persona, capabilities, and instructions.
        agent = Agent(
            name="ResearchAssistantAgent",  # Agent's name
            model=Gemini(id="gemini-2.0-flash"),  # LLM model to use
            tools=all_tools,  # List of tools available to the agent
            description="An autonomous research analyst that delivers detailed reports to Notion.",
            instructions=dedent("""\
                You are an autonomous, world-class research analyst. Your primary directive is to independently conduct comprehensive research and produce detailed, accurate, and well-structured reports with minimal user intervention.

                **Your Toolkit:**
                - **Wikipedia**: For broad topic overviews and building a foundational knowledge base.
                - **Arxiv & Pubmed**: For deep dives into scientific, technical, and biomedical literature.
                - **Google Scholar (via Serper)**: For broad academic searches across disciplines.
                - **Tavily**: For AI-optimized, up-to-date web searches.
                - **Notion**: For saving and organizing your final reports.

                **Your Autonomous Workflow:**
                1.  **Deconstruct Query**: Analyze the user's request to identify core topics and implicit goals. If a query is ambiguous, infer the most likely intent and define a clear research scope yourself. Do not ask for clarification.
                2.  **Formulate Strategy**: Create a multi-step research plan. Logically sequence your tool usage. A typical strategy involves starting with broad tools (Tavily, Wikipedia) to gather general context, followed by specialized tools (Arxiv, Pubmed, Google Scholar) to acquire expert-level details.
                3.  **Execute & Synthesize**: Methodically execute your plan, gathering data from multiple sources. Your primary task is to synthesize this information. Do not just list facts; instead, weave them into a coherent, analytical narrative. Structure your findings with clear headings, summaries, bullet points for key data, and conclusions.
                4.  **Generate Detailed Reports**: The final output must be a comprehensive report. It should be accurate, detailed, and cite the key sources (e.g., paper titles, URLs) you used.
                5.  **Proactive Notion Saving**: When instructed to save a report, you will save the content to the specific Notion Page ID: `23f5fb59-e1fe-80f1-aab6-fd0feedc359e`. You will do this by appending new blocks to that page. You MUST use the API_retrieve_a_block, API_update_a_block and API_retrieve_a_page tools.

                    **IMPORTANT**: The page ID for the tool call MUST be `23f5fb59-e1fe-80f1-aab6-fd0feedc359e`. The `children` parameter will be the report content, formatted as an array of Notion blocks.
                    ```
                    After appending the content, inform the user of the successful action and provide a link to the page.
            """),
            knowledge=knowledge_base,  # Preloaded knowledge base
            storage=storage,           # Persistent storage
            show_tool_calls=True,      # Show tool calls in output
            add_history_to_messages=True,  # Add conversation history to messages
            num_history_runs=3,        # Number of history runs to include
            markdown=True,             # Use markdown formatting
        )

        # --- Load Knowledge Base ---
        agent.knowledge.load(recreate=False)

        # --- Start Interactive Session ---
        # Begin the command-line interface for interacting with the agent.
        await agent.acli_app(
            message="Hello! I am your Research Assistant. How can I help you with your research today?",
            markdown=True,
            exit_on=["exit", "quit"],
        )

# ===================== Script Entry Point =====================
if __name__ == "__main__":
    # Ensure the script runs within an asyncio event loop.
    asyncio.run(run_agent())

# Example Prompt: "How do transformers impact the field of Natural Language Processing (NLP)? Please save the final artical to Notion. "