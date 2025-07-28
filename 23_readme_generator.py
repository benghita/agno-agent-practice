from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.github import GithubTools
from agno.tools.local_file_system import LocalFileSystemTools
from dotenv import load_dotenv

load_dotenv()

readme_gen_agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    tools=[GithubTools(), LocalFileSystemTools()],
    markdown=True,
    debug_mode=True,
    description="You are an intelligent assistant that analyzes the contents of a GitHub repository and automatically generates a high-quality `README.md` file.",
    instructions=[
        "- Use get_repository to access to the given repo then:",
        "  - Fetch repo metadata (name, description, topics, license)",
        "  - Analyze project structure (folders, files, tech stack)",
        "  - Detect main entry points, usage patterns, and dependencies",
        "  - Extract key content from `package.json`, `pyproject.toml`, or config files",
        " - Organize the README using the following standard sections:",
        "  1. **Project Title & Description**",
        "  2. **Badges** (e.g., build status, license, version)",
        "  3. **Features** (optional)",
        "  4. **Installation**",
        "  5. **Usage**",
        "  6. **Configuration** (if applicable)",
        "  7. **Contributing**",
        "  8. **License**",
        "  9. **Acknowledgments** (optional)",
    ],
)

readme_gen_agent.print_response(
    "Get details of https://github.com/agno-agi/agno", markdown=True
)