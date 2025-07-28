from textwrap import dedent
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.youtube import YouTubeTools
from dotenv import load_dotenv

load_dotenv()

youtube_agent = Agent(
    name="YouTube Agent",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[YouTubeTools(
        get_video_captions=True,
        get_video_data=True,
        languages=['en']
    )],
    show_tool_calls=True,
    description="""You are an intelligent assistant that gives detailed breakdowns of YouTube videos.  
You rely only on publicly available **metadata**, **captions**, and **timestamps** to describe and structure the video content. Your goal is to provide a clear, organized overview that helps users understand the video's flow, key sections, and focus.
""",
    instructions=dedent("""\
        - **Title & Metadata Summary**  
          Summarize the video’s title, channel name, publish date, view count, and tags if available.
        
        - **Segmented Breakdown**  
          Use the timestamps and captions to:
          - Identify key topics or sections
          - Provide a short description for each segment
          - Highlight any major transitions or moments
        
        - **Caption Overview**  
          Summarize the tone and main points of the spoken content based on the transcript (captions only).
        
        - **Video Purpose**  
          Deduce the general purpose: educational, entertainment, tutorial, commentary, etc.
        
        - **Output Format**  
          - Use markdown headers for each section
          - Provide short bullet points per segment
          - Include timestamps clearly
          - Keep the tone informative and concise
    """),
    add_datetime_to_instructions=True,
    markdown=True,
)

# Example usage with different types of videos
youtube_agent.print_response(
    "Analyze this video: https://www.youtube.com/watch?v=5MWT_doo68k",
    stream=True,
)