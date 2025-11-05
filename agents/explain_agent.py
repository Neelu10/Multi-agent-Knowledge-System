from crewai import Agent
from openai import OpenAI
from llm_setup import llm
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
client = OpenAI(api_key=api_key, base_url=base_url)


explain_agent = Agent(
    role="Explanation Agent",
    goal="Simplify and refine the domain expert’s answer into user-friendly language.",
    backstory=(
        "You are a communication specialist who rewrites technical or policy-heavy responses "
        "into clear, conversational, and easy-to-understand explanations for employees."
    ),
    llm=llm,
)

def simplify_answer(answer: str) -> str:
    """
    Simplifies and humanizes an answer — whether from FAQ or LLM.
    Automatically switches to OpenAI API if CrewAI llm fails.
    """
    if not answer or "not found" in answer.lower():
        return "Sorry, I couldn’t find that information right now."

    prompt = (
        "Please rewrite the following answer in simple, friendly, and conversational language "
        "while keeping all important details:\n\n" + answer
    )


    try:
        response = explain_agent.run(prompt)
        if response and len(response.strip()) > 0:
            return response.strip()
    except Exception as e:
        print("[WARN] CrewAI simplification failed:", e)

    # Step 2: Fallback to OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("[ERROR in simplify_answer fallback]", e)
        return answer.strip()
