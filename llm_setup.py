from crewai import LLM
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Detect key type
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not api_key:
    raise ValueError("❌ No API key found. Add OPENROUTER_API_KEY or OPENAI_API_KEY in .env")

# ✅ Create LLM for CrewAI
llm = LLM(model="gpt-3.5-turbo", api_key=api_key, base_url=base_url)

# ✅ Create OpenAI client for custom use
client = OpenAI(api_key=api_key, base_url=base_url)
