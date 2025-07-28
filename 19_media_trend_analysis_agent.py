from textwrap import dedent
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.exa import ExaTools
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    tools=[
        ExaTools(type="keyword"),
        DuckDuckGoTools(),
    ],
    description=dedent("""\
        You are an expert AI assistant that analyzes media content from news outlets, social media, and digital platforms
    """),
    instructions=[
        "- Continuously scan and analyze real-time or recent media data (news headlines, articles, tweets, forums, etc.).",
        "- Compare current patterns to historical baselines to detect shifts.",
        "- Synthesize the information into clear insights, backed by quantitative or qualitative reasoning.",
        "- When forecasting, use current trends, sentiment, and volume signals.",
        "- Always ground your insights in the observed data — no speculation without evidence.",
    ],
    expected_output=dedent("""\
    ### 1. **Trend Summary**
    - Brief title or name of the identified trend.
    
    ### 2. **Description**
    - A concise explanation of the trend, including what is new or significant.
    
    ### 3. **Supporting Data**
    - Key stats, quotes, article headlines, or engagement metrics.
    
    ### 4. **Pattern Change**
    - Describe how this differs from previous patterns (volume, sentiment, narrative shifts, etc.).
    
    ### 5. **Actionable Insight**
    - What can stakeholders do with this information? (e.g., PR planning, investment focus, public awareness)
    
    ### 6. **Forecast**
    - A short prediction of how this trend might evolve over the next weeks or months.
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)

agent.print_response("Analyze media trends for AI agent in linkedin and X", stream=True)
