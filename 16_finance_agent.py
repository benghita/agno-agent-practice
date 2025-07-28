from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
from textwrap import dedent

load_dotenv()

financial_analyst = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    description="You are a professional financial analyst providing in-depth, real-time market insights.",
    instructions=dedent(
        """
        - Use **real-time data from Yahoo Finance** to generate reports.
        - Combine **quantitative data**, **analyst insights**, and **market context**.
        - Organize the report into these sections:
        
        1. **Company Overview**  
           - Company description, sector, market cap, ticker.
        
        2. **Financial Highlights**  
           - Key metrics (revenue, EPS, ratios), growth trends.
        
        3. **Stock Performance**  
           - Price trends, volatility, volume.
        
        4. **Analyst Ratings & Forecasts**  
           - Analyst recommendations, target prices, earnings forecasts.
        
        5. **Market Context & News**  
           - Recent news, industry and macro trends, competitor info.
        
        6. **Outlook & Summary**  
           - Financial health, risks, opportunities, overall assessment.
        
        - Reports should be clear and well-structured.
        - Include charts or tables if helpful.
        """
    ),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            historical_prices=True,
            company_info=True,
            company_news=True,
            income_statements=True,
            key_financial_ratios=True,
            technical_indicators=True
        ), 
        DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

financial_analyst.print_response("Analyze Apple Inc. and write a full report using the latest data.")