from crewai import Agent
from openai import OpenAI
from llm_setup import llm
from dotenv import load_dotenv
import os, json

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
client = OpenAI(api_key=api_key, base_url=base_url)


router_agent = Agent(
    role="Router Agent",
    goal="Identify which department (HR, Finance, or IT) the user's question belongs to.",
    backstory=(
        "You are a smart classifier who reads a user's question and determines "
        "which department is responsible for answering it. "
        "HR handles leaves, holidays, policies, attendance, etc. "
        "Finance manages salaries, reimbursements, invoices, and tax queries. "
        "IT deals with email setup, password resets, and technical issues."
    ),
    llm=llm,
)


def classify_question(question: str) -> str:
    """
    Classify a question into HR, Finance, or IT.
    Falls back to OpenAI API if CrewAI fails.
    """
    if not question.strip():
        return "HR"

    prompt = (
        f"Classify this question into one of these categories: HR, Finance, or IT.\n"
        f"Question: {question}\n"
        f"Respond ONLY with one word: HR, Finance, or IT."
    )


    try:
        response = router_agent.run(prompt)
        response_text = str(response).strip()
        # Normalize output (case-insensitive)
        if "finance" in response_text.lower():
            return "Finance"
        elif "it" in response_text.lower():
            return "IT"
        else:
            return "HR"
    except Exception as e:
        print("[WARN] CrewAI classification failed:", e)


    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        if "finance" in answer.lower():
            return "Finance"
        elif "it" in answer.lower():
            return "IT"
        else:
            return "HR"
    except Exception as e:
        print("[ERROR in classify_question fallback]", e)
        return "HR"
