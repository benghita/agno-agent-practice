from pathlib import Path
from textwrap import dedent
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.cartesia import CartesiaTools
from agno.tools.reasoning import ReasoningTools
from agno.utils.media import download_file, save_base64_data
from dotenv import load_dotenv
import requests
import os

load_dotenv()
api_key = os.getenv("CARTESIA_API_KEY")

input_audio_url: str = (
    "https://agno-public.s3.us-east-1.amazonaws.com/demo_data/sample_audio.mp3"
)

local_audio_path = Path("tmp/meeting_recording.mp3")
print(f"Downloading file to local path: {local_audio_path}")
download_file(input_audio_url, local_audio_path)

def speech_to_text(audio_file_path: str, api_key: str = api_key, language: str = "en") -> dict:
    """
    Convert speech from an audio file to text using Cartesia API.
    
    Args:
        audio_file_path (str): Path to the audio file to transcribe
        api_key (str): Cartesia API key for authentication
        language (str, optional): Language code for transcription. Defaults to "en".
    
    Returns:
        dict: JSON response from the Cartesia API containing the transcription
        
    Raises:
        FileNotFoundError: If the audio file doesn't exist
        requests.RequestException: If the API request fails
    """
    cartesia_url = "https://api.cartesia.ai/stt"
    
    # Check if file exists
    if not Path(audio_file_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    # Prepare the request
    with open(audio_file_path, 'rb') as audio_file:
        files = {"file": audio_file}
        payload = {
            "model": "ink-whisper",
            "language": language,
        }
        headers = {
            "Cartesia-Version": "2024-11-13",
            "X-API-Key": api_key
        }
        
        response = requests.post(cartesia_url, data=payload, files=files, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes
        
        return response.json()


meeting_agent: Agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    tools=[ReasoningTools(), CartesiaTools(), speech_to_text],
    description=dedent("""\
        An AI agent that processes audio recordings of meetings to:
        1. **Extract key information** (decisions, tasks, participants, insights).
        2. **Create a visual representation** (e.g., timeline, topic map, task chart).
        3. **Generate an audio summary** for quick playback.
    """),
    instructions=dedent(f"""\
        - Transcribe the audio using the speech_to_text tool.
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

if __name__ == "__main__":

    response = meeting_agent.run(
        f"Please process the meeting recording located at '{local_audio_path}'",
    )
    print(response)
    if response.audio:
        save_base64_data(response.audio[0].base64_audio, Path("tmp/meeting_summary.mp3"))
        print(f"Meeting summary saved to: {Path('tmp/meeting_summary.mp3')}")