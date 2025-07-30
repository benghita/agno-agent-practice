# Import necessary libraries for the recipe visualization system
from pathlib import Path
from agno.agent import Agent
from agno.team.team import Team
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.tools.thinking import ThinkingTools
from agno.models.google import Gemini
from agno.utils.media import download_image
from agno.vectordb.lancedb import LanceDb
from agno.tools.replicate import ReplicateTools
from dotenv import load_dotenv
from textwrap import dedent
import replicate

# Load environment variables (API keys, etc.)
load_dotenv()

def generate_images_with_replicate(
    descriptions: list[str],
    model_name: str = "black-forest-labs/flux-schnell",
) -> dict[str, str]:
    """
    Generate images from a list of text descriptions using Replicate API.
    
    This function takes multiple recipe step descriptions and generates corresponding
    images using the Replicate AI image generation service.

    Parameters:
        descriptions (list[str]): List of text descriptions for image generation.
        model_name (str): The Replicate model to use. Default is 'black-forest-labs/flux-schnell'.
    Returns:
        dict[str, str]: Dictionary mapping descriptions to image URLs.
    """
    
    # Dictionary to store description -> image URL mappings
    result = {}
    
    try:
        # Process each description in the list
        for i, description in enumerate(descriptions):
            print(f"Generating image {i+1}/{len(descriptions)}: {description[:50]}...")
            
            # Call Replicate API to generate image from text description
            output = replicate.run(
                model_name,
                input={
                    "prompt": description,           # Text description to convert to image
                    "num_outputs": 1,                # Generate 1 image per description
                    "seed": 12345,                   # Fixed seed for reproducible results
                    "megapixels": "1",               # Image resolution (1 megapixel)
                    "aspect_ratio": "1:1",           # Square aspect ratio
                    "output_format": "jpg",          # Output format
                    "output_quality": 80,            # Image quality (1-100)
                    "num_inference_steps": 4         # Number of generation steps (faster generation)
                }
            )
            
            # Extract the image URL from the API response
            if output and len(output) > 0:
                # Replicate returns a list where the first element is the image URL
                image_url = output[0] 
                result[description] = image_url
                print(f"Successfully generated image {i+1}: {image_url}")
            else:
                print(f"Failed to generate image for description {i+1}")
                result[description] = None
                
    except Exception as e:
        print(f"Image generation failed: {e}")
        # Return partial results if some images were generated successfully
        return result
    
    return result


# Initialize knowledge base with Thai recipes PDF
# This creates a searchable database of recipe information
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=LanceDb(
        uri="tmp/lancedb",                           # Local database storage location
        table_name="embed_vision_documents",         # Table name for storing embeddings
        embedder=GeminiEmbedder(),                   # Use Google's Gemini for text embedding
    ),
)
knowledge_base.load()  # Load and process the PDF content

RecipeVisualizerAgent = Agent(
    role="Take each of the five recipe steps and generate a vivid image description for it, then produce a realistic or stylized image using a replicate API.",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[generate_images_with_replicate],
    description="A visual content creator that takes textual cooking instructions and turns them into stunning, step-by-step image representations using generate_images_with_replicate.",
    instructions=dedent(
        """You receive step-by-step cooking instructions from another agent.
           
           Your job is to:
           1. Interpret each step’s instruction and convert it into a vivid, descriptive **visual prompt**.
           2. Use "generate_images_with_replicate" tool one time, give it the list of prompts to create one **realistic or stylized image** per step the toole will return a dictionary of the steps with images urls.
           
           🎨 **Guidelines for Visual Prompts**:
           - Include rich detail: ingredients, cooking tools, food textures, colors, lighting.
           - Mention perspective (e.g., top view, side view), setting (e.g., kitchen counter, stove), and action (e.g., chopping, stirring).
           - Maintain stylistic consistency across all steps (realistic OR stylized, not mixed).
           - Do not include text in images.
           
           ✅ **Output Format**:
           {
           `Step 1 Description`: `Generated Image URL 1`,
           `Step 2 Description`: `Generated Image URL 2`,
            ...
           }
           Make sure to return the images urls not the image objects"""
    ),
    markdown=True,
    show_tool_calls=True,
)

# Main team agent that orchestrates recipe simplification and visualization
RecipeSimplifierAgent = Team(
    members=[RecipeVisualizerAgent],                 # Include the visualizer agent as a team member
    model=Gemini(id="gemini-2.0-flash"),            # Use Gemini for team coordination
    tools=[ThinkingTools()],                         # Enable reasoning and analysis tools
    knowledge=knowledge_base,                        # Access to recipe database
    description="You are a world-class culinary assistant.",
    instructions=dedent(
        """Your tasks:
        1. Retrieve a full recipe from a cookbook or reliable source using available tools.
        2. Analyze and simplify it into **exactly five key steps**, each clearly named and described in 1–2 sentences.
        3. Send the 5 steps (text only) to the `RecipeVisualizerAgent` for image generation.
        4. Wait for the response: For each step, you will receive:
           - An image generation prompt.
           - A generated image or reference to it.
        5. **Combine** the text and image of each step into one structured Markdown block.

        🧾 **Final Output Format**:
        ```markdown
        ## Recipe Title: [Insert Recipe Name]

        ### Step 1: [Step Title]
        **Instruction**: [Your simplified step instruction]  
        **Image Prompt**: _[Prompt used to generate image]_  
        ![Step 1](image_url_or_placeholder)

        ...repeat for steps 2–5... """
    ),
    markdown=True,                                    # Enable markdown formatting
    show_members_responses=True,                     # Show responses from team members
)

# Execute the recipe agent system with a sample query
RecipeSimplifierAgent.print_response(
    "Teach me how to make Papaya Salad.",
)

# Get the response and optionally download any generated images
response = RecipeSimplifierAgent.run_response
if response.images:
    download_image(response.images[0].url, Path("tmp/recipe_image.png"))