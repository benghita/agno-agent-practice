# ===================== Imports =====================
# agno-agent framework imports
from agno.agent import Agent  # Main Agent class
from agno.team.team import Team  # Team coordination class
from agno.models.google import Gemini  # Google Gemini LLM model
from agno.knowledge.url import UrlKnowledge  # Knowledge base from URLs
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase  # Knowledge base from PDF URLs
from agno.knowledge.combined import CombinedKnowledgeBase  # Combined knowledge sources
from agno.tools.visualization import VisualizationTools  # Chart creation tools
from agno.tools.pandas import PandasTools  # Data analysis tools
from agno.embedder.google import GeminiEmbedder  # Embedding model for vector DB
from agno.storage.sqlite import SqliteStorage  # SQLite-based storage for sessions
from agno.tools.duckdb import DuckDbTools  # Database query tools
from agno.vectordb.lancedb import LanceDb, SearchType  # Vector DB with search types
from agno.tools.file import FileTools  # File operations tools
from dotenv import load_dotenv  # For loading environment variables
from textwrap import dedent  # For formatting multi-line strings

# ===================== Load Environment Variables =====================
load_dotenv()  # Loads variables from a .env file into environment

# ===================== Knowledge Base Setup =====================
# PDF Knowledge Base - Loads climate policy documents
pdf_url_kb = PDFUrlKnowledgeBase(
    urls=[
        "https://www.horizont3000.org/media/pages/topics/climate-action/policy/eae40da3a5-1725008907/h3-environmental-climate-policy-2023-en.pdf",
    ],
    vector_db=LanceDb(
        table_name="environmental-climate-policy", 
        uri="tmp/lancedb",
        embedder=GeminiEmbedder(),
    )
)

# Website Knowledge Base - Loads climate change information from Royal Society
website_kb = UrlKnowledge(
    urls=["https://royalsociety.org/news-resources/projects/climate-change-evidence-causes/basics-of-climate-change/"],
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="climate-change",
        search_type=SearchType.hybrid,  # Use hybrid search for better results
        embedder=GeminiEmbedder(),
    ),
)

# Combined Knowledge Base - Merges multiple sources into one searchable base
knowledge_base = CombinedKnowledgeBase(
    sources=[
        pdf_url_kb,
        website_kb
    ],
    vector_db=LanceDb(
        table_name="climate_change_base", 
        uri="tmp/lancedb",
        embedder=GeminiEmbedder(),
    )
)

# ===================== Storage Setup =====================
# Configure persistent storage for agent sessions using SQLite
storage = SqliteStorage(
    table_name="sessions",
    db_file="tmp/agent.db"
)

# ===================== Knowledge Researcher Agent =====================
# This agent specializes in synthesizing scientific and policy documents
knowledge_researcher = Agent(
    role="""Expert Knowledge Researcher specialized in synthesizing scientific and policy documents into comprehensive, 
            well-structured reports with clear sections and proper citations.""",
    model=Gemini(id="gemini-2.0-flash"),  # LLM model to use
    description=dedent("""
        Retrieves and synthesizes high-quality scientific and policy-related information from vector databases
        Creates detailed, structured articles that are human-readable and evidence-based
        Organizes content logically with proper flow and citations
        Provides comprehensive background context for data visualization needs
        """),
    instructions=dedent("""
        RESEARCH PROCESS:
        1. Use RAG to retrieve the top 10 most relevant chunks for comprehensive coverage
        2. Synthesize retrieved content into an original, well-structured explanation
        3. Organize your output using these mandatory sections:
           - **Executive Summary** (2-3 sentences overview)
           - **Historical Context & Evidence** (detailed background with timeline)
           - **Current State & Key Findings** (present situation with specific data points)
           - **Policy Implications & Recommendations** (actionable insights)
           - **Data Requirements for Visualization** (specify what charts would be most valuable)
           - **Source References** (brief quotes with chunk references)
        
        QUALITY REQUIREMENTS:
        - Write in clear, accessible language suitable for policy makers and educated public
        - Include specific numbers, dates, and quantitative information when available
        - Maintain logical flow from historical context to current state to future implications
        - Clearly identify gaps where data visualization would enhance understanding
        - Use markdown formatting for structure and readability
        """),
    knowledge=knowledge_base,  # Access to combined knowledge sources
    markdown=True  # Use markdown formatting
)

# Load the knowledge base into memory
knowledge_researcher.knowledge.load(recreate=False)

# ===================== Data Analyst & Visualizer Agent =====================
# This agent specializes in data analysis and chart creation
analyst_visualizer = Agent(
    role="""Senior Data Analyst & Visualization Specialist focused on creating compelling, publication-ready charts 
            that support climate research and policy analysis.""",
    model=Gemini(id="gemini-2.0-flash"),  # LLM model to use
    instructions=dedent("""
        DATA ANALYSIS WORKFLOW:
        1. **Data Discovery**: Use DuckDbTools to systematically search for all available CSV files with climate-related data
        2. **Context Integration**: Carefully read the research context provided by the Knowledge Researcher
        3. **Data Selection**: Choose the most relevant datasets that directly support the research findings
        4. **Analysis & Visualization**: 
           - Use PandasTools for thorough data analysis and preprocessing
           - Create multiple complementary visualizations using VisualizationTools
           - Generate at least 2-3 different chart types (line charts for trends, bar charts for comparisons, etc.)
        
        CHART CREATION REQUIREMENTS:
        - Save ALL charts as high-resolution JPG files with descriptive filenames
        - Use format: "chart_[topic]_[type]_[date].jpg" (e.g., "chart_co2_emissions_trend_2024.jpg")
        - Include chart titles, axis labels, legends, and data source citations
        - Ensure charts are publication-ready with professional styling
        
        OUTPUT FORMAT (MANDATORY):
        For each chart created, provide:
        ```
        **Chart [Number]: [Descriptive Title]**
        - File Path: [full path to saved JPG file]
        - Chart Type: [line/bar/scatter/etc.]
        - Data Source: [CSV filename and relevant columns]
        - Key Insights: [2-3 bullet points of main findings]
        - Methodology: [brief description of data processing]
        ```
        
        FINAL DELIVERABLE:
        - End your response with a complete list of all generated chart file paths
        - Include a summary of how the visualizations support the research narrative
        - Provide data interpretation that connects charts to policy implications
        """),
    tools=[DuckDbTools(), VisualizationTools(), PandasTools()],  # Data analysis and visualization tools
    show_tool_calls=True,  # Show tool calls in output
    markdown=True  # Use markdown formatting
)

# ===================== Team Leader Agent =====================
# This agent coordinates the work between the knowledge researcher and data analyst
leader_agent = Team(
    members=[analyst_visualizer, knowledge_researcher],  # Team members
    model=Gemini(id="gemini-2.0-flash"),  # LLM model for team coordination
    description="""Executive Team Leader responsible for coordinating comprehensive climate analysis reports 
        that integrate research findings with data visualizations into professional, actionable documents.""",
    instructions=dedent("""
        COORDINATION PROCESS:
        1. **Task Delegation**: 
           - First, send the user query to Knowledge Researcher for comprehensive research synthesis
           - Then, send the research context + user query to Data Analyst for visualization creation
           - Ensure both agents understand they're working on the same integrated report
        
        2. **Quality Control & Integration**:
           - Review both outputs for completeness and coherence
           - Verify that all generated charts are properly documented with file paths
           - Ensure visualizations directly support and enhance the research findings
           - Check that data interpretations align with research conclusions
        
        3. **REPORT COMPILATION** (Save using FileTools):
           Create a comprehensive report file named "climate_analysis_report_[timestamp].txt" containing:
           
           ```
           COMPREHENSIVE CLIMATE ANALYSIS REPORT
           Generated: [current date and time]
           =====================================
           
           EXECUTIVE SUMMARY
           [Key findings in 3-4 sentences]
           
           DETAILED RESEARCH FINDINGS
           [Complete research output from Knowledge Researcher]
           
           DATA ANALYSIS & VISUALIZATIONS
           [Complete analysis output from Data Analyst]
           
           CHART INVENTORY
           [List all chart file paths with descriptions]
           
           INTEGRATED INSIGHTS
           [How visualizations support research findings]
           
           POLICY RECOMMENDATIONS
           [Actionable recommendations based on combined research and data]
           
           APPENDICES
           - Data Sources Used
           - Methodology Summary
           - Chart File References
           ```
        
        4. **VERIFICATION CHECKLIST** (Include in response):
           - ✓ Research section complete with citations
           - ✓ All requested visualizations created and saved
           - ✓ Chart file paths documented and accessible
           - ✓ Report saved with integrated content
           - ✓ Data interpretations align with research findings
           - ✓ Actionable recommendations provided
        
        CRITICAL REQUIREMENTS:
        - The final report MUST include the actual file paths of all generated charts
        - Charts must be referenced within the research narrative, not just listed
        - Use FileTools to save the complete integrated report
        - Provide the final report file path in your response
        - Ensure the report is self-contained and professional
        """),
    tools=[FileTools()],  # File operations tools
    storage=storage,  # Persistent storage for team sessions
    show_tool_calls=True,  # Show tool calls in output
    show_members_responses=True,  # Show responses from team members
    markdown=True  # Use markdown formatting
)

# ===================== Execute Team Analysis =====================
# Run the team analysis on climate change and CO₂ emissions
leader_agent.print_response("Explain the main contributors to global CO₂ emissions and how they have changed since the industrial revolution. Base the explanation on scientific sources and policies. give a full report with visualisations.")