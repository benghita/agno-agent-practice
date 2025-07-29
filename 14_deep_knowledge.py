from agno.agent import Agent
from agno.team.team import Team
from agno.models.google import Gemini
from agno.knowledge.url import UrlKnowledge
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.tools.visualization import VisualizationTools
from agno.tools.pandas import PandasTools
from agno.embedder.google import GeminiEmbedder
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckdb import DuckDbTools
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.tools.file import FileTools
from dotenv import load_dotenv
from textwrap import dedent

load_dotenv()

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

website_kb = UrlKnowledge(
    urls=["https://royalsociety.org/news-resources/projects/climate-change-evidence-causes/basics-of-climate-change/"],
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="climate-change",
        search_type=SearchType.hybrid,
        embedder=GeminiEmbedder(),
    ),
)

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

storage = SqliteStorage(
    table_name="sessions",
    db_file="tmp/agent.db"
)

knowledge_researcher = Agent(
    role="""Expert Knowledge Researcher specialized in synthesizing scientific and policy documents into comprehensive, 
            well-structured reports with clear sections and proper citations.""",
    model=Gemini(id="gemini-2.0-flash"),
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
    knowledge=knowledge_base,
    markdown=True
)

knowledge_researcher.knowledge.load(recreate=False)

analyst_visualizer = Agent(
    role="""Senior Data Analyst & Visualization Specialist focused on creating compelling, publication-ready charts 
            that support climate research and policy analysis.""",
    model=Gemini(id="gemini-2.0-flash"),
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
    tools=[DuckDbTools(), VisualizationTools(), PandasTools()],
    show_tool_calls=True,
    markdown=True
)

leader_agent = Team(
    members=[analyst_visualizer, knowledge_researcher],
    model=Gemini(id="gemini-2.0-flash"),
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
    tools=[FileTools()],
    storage=storage,
    show_tool_calls=True,
    show_members_responses=True,
    markdown=True
)

leader_agent.print_response("Explain the main contributors to global CO₂ emissions and how they have changed since the industrial revolution. Base the explanation on scientific sources and policies. give a full report with visualisations.")