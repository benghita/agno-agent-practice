from pathlib import Path
from textwrap import dedent
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.openai import OpenAITools
from agno.tools.reasoning import ReasoningTools
from agno.utils.media import download_file, save_base64_data
from dotenv import load_dotenv

load_dotenv()

input_audio_url: str = (
    "https://agno-public.s3.us-east-1.amazonaws.com/demo_data/sample_audio.mp3"
)

local_audio_path = Path("tmp/meeting_recording.mp3")
print(f"Downloading file to local path: {local_audio_path}")
download_file(input_audio_url, local_audio_path)

meeting_agent: Agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    tools=[ReasoningTools(), OpenAITools()],
    description=dedent("""\
        An AI agent that processes audio recordings of meetings to:
        1. **Extract key information** (decisions, tasks, participants, insights).
        2. **Create a visual representation** (e.g., timeline, topic map, task chart).
        3. **Generate an audio summary** for quick playback.
    """),
    instructions=dedent(f"""\
        - Transcribe the audio using accurate speech-to-text.
        - Identify and extract:
            - Key topics discussed
            - Decisions made
            - Action items (who, what, when)
            - Notable quotes or concerns
        - Organize the extracted data into a structured summary.
        - Generate:
            - A **visual summary** (timeline, mind map, or flow of discussion).
            - A **concise audio summary** (< 2 minutes) using TTS.
    """),
    markdown=True,
    show_tool_calls=True,
)

response = meeting_agent.run(
    f"Please process the meeting recording located at '{local_audio_path}'",
)
print(response)
if response.audio:
    save_base64_data(response.audio[0].base64_audio, Path("tmp/meeting_summary.mp3"))
    print(f"Meeting summary saved to: {Path('tmp/meeting_summary.mp3')}")