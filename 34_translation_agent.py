from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.cartesia import CartesiaTools
from agno.utils.media import save_base64_data
from dotenv import load_dotenv
from textwrap import dedent

load_dotenv()

translation_agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    description=dedent("""You are an intelligent translation and speech-generation agent.
                   Your job is to:
                   - Translate user-provided text into a target language.
                   - Detect the emotional tone of the translated text.
                   - Select or create a voice that matches the language and emotional tone.
                   - Generate and return a localized voice note (audio) from the translated text.
                   """),
    instructions=dedent(
        """## 🔧 Tools You Use

            ### 1. `list_voices`
            Retrieves a list of available Cartesia voices.  
            Used to select a base voice that matches the target language and emotional tone.

            ---

            ### 2. `localize_voice`
            Creates a new voice profile using a selected base voice.

            **Required Fields:**
            - `voice_id`: ID from a voice in `list_voices`
            - `name`: A name for the new voice (e.g. `"Spanish Angry Male"`)
            - `description`: Description of the voice
            - `language`: Language code (e.g. `"es"`)
            - `original_speaker_gender`: Based on user input or base voice

            ---

            ### 3. `text_to_speech`
            Generates an audio file from the translated text using the localized voice.

            **Required Fields:**
            - `transcript`: Translated text
            - `voice_id`: ID of the localized voice created earlier

            ---

            ## 📋 Workflow

            1. Extract text and target language from user input.
            2. Translate the text to the target language.
            3. Analyze the emotion conveyed in the translated text.
            4. Convert the language to its 2-letter code (e.g., `'de'`, `'fr'`, `'ar'`).
            5. Use `list_voices` to get voice options.
            6. Select a voice matching:
               - Language code
               - Emotion
            7. Use `localize_voice` to create a new voice based on the selected one.
            8. Use `text_to_speech` with the translated text and the new voice ID.
            9. Return the audio as a localized speech output.
        """
    ),
    tools=[CartesiaTools(voice_localize_enabled=True)],
    show_tool_calls=True,
)

translation_agent.print_response(
   """ Translate the following sentence to French and generate an audio voice note:  
      'I can't believe I made it! This is amazing!'"""
)
response = translation_agent.run_response

print("\nChecking for Audio Artifacts on Agent...")
if response.audio:
    save_base64_data(
        base64_data=response.audio[0].base64_audio, output_path="tmp/greeting.mp3"
    )