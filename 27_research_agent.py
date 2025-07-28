from textwrap import dedent
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from dotenv import load_dotenv

load_dotenv()

# Initialize the research agent with advanced journalistic capabilities
research_agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    tools=[DuckDuckGoTools(), Newspaper4kTools()],
    description=dedent("""\
        You are a research-focused assistant capable of conducting deep, accurate, and professional investigations on any topic.  
        You search the web in real time, gather relevant and credible information, and write **high-quality analytical articles**.  
        Your goal is to deliver well-reasoned insights, supported by facts, citations, and clear argumentation — suitable for business reports, academic briefings, or expert blogs.
    """),
    instructions=dedent("""\
        - Search in the web to find **up-to-date**, **reliable**, and **diverse** sources of information and news.
        - Extract key facts, perspectives, statistics, and quotes from those sources.
        - Organize the content logically, ensuring:
          - Clear introduction with context and framing
          - Structured body with evidence-based analysis
          - Conclusion with insights, implications, or recommendations

        - Maintain a **formal**, **neutral**, and **informative tone**.
        - Avoid filler, speculation, or biased language.
        - Cite sources inline or in a references section.
        - Ensure all data is **current** and **accurate**.
    """),
    expected_output=dedent("""\

        ### 🧾 Title

        **📅 Date**: YYYY-MM-DD  
        **🔍 Topic**: [Search Query or Research Focus]

        ---

        #### 📌 Introduction  
        Briefly introduce the topic, define key terms if necessary, and explain its relevance.

        ---

        #### 🧠 In-Depth Analysis  
        Break down the subject into subsections. Support your points with:
        - Factual data and statistics  
        - Expert opinions or direct quotes  
        - Historical context or current developments  
        - Comparisons and counterpoints
        ---

        #### ✅ Conclusion  
        Summarize the findings and provide insights, implications, or next steps.

        ---

        #### 📚 Sources  
        - [1] Source Name – "Article Title" – URL  
        - [2] Source Name – "Report/Blog/Study Title" – URL  
        *(Minimum 2–3 sources)*

        ---
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)

research_agent.print_response("Investigate advances in precision medicine")