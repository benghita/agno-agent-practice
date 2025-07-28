from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from urllib.parse import urlparse
from dotenv import load_dotenv
from textwrap import dedent
from bs4 import BeautifulSoup
import httpx
import json

load_dotenv()

def scrape_text_from_url(url: str, max_length: int = 3000) -> str:
    """
    Use this tool to extract readable text content from a given webpage URL.

    Args:
        url (str): The URL to scrape.
        max_length (int): Max number of characters to return. Defaults to 3000.

    Returns:
        str: JSON string with extracted text or error message.
    """
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return json.dumps({"error": "Invalid URL."})

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)",
        }
        response = httpx.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted elements
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=' ', strip=True)

        if len(text) > max_length:
            text = text[:max_length] + "..."

        return json.dumps({"text": text})

    except httpx.RequestError as e:
        return json.dumps({"error": f"Request error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})

agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    description="You are a research assistant focused on competitor analysis. You help users gather and summarize useful information about competing companies or products using web search and scraping tools.",
    instructions=[
        dedent("""
            1. **Understand the user's goal**: Identify which company, product, or market segment they want to analyze.

            2. **Search the web**:
               - Use `duckduckgo_search` to find recent or relevant pages related to the competitor.
               - Focus on official websites, product comparison pages, customer reviews, and news articles.

            3. **Extract content**:
               - For the most relevant URLs, use `scrape_text_from_url` to extract main text content.
               - Summarize key information (e.g., product features, pricing, positioning, strengths/weaknesses).

            4. **Deliver a detailed report**:
               - Present the important insights clearly and concisely.
               - Include company names, offerings, differentiators, and strategic moves.
            
            5. **Handle errors gracefully**:
               - If `scrape_text_from_url` returns an error, log or report the error.
               - If the page couldn't be scraped, try a different URL or summarize based on the search snippet or another source.
        """)
    ],
    tools=[scrape_text_from_url, 
           DuckDuckGoTools(),
           ],
    show_tool_calls=True,
    markdown=True
)

agent.print_response("Analyze the competitive landscape for Stripe in the payments industry.", stream=True)
