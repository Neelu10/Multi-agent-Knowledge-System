from crewai import Agent
from difflib import SequenceMatcher
from llm_setup import llm
import os


hr_agent = Agent(
    role="HR Expert",
    goal="Answer company HR-related queries accurately and politely.",
    backstory=(
        "You are an HR specialist responsible for answering employee questions "
        "related to leaves, policies, attendance, recruitment, holidays, etc."
    ),
    llm=llm,
)

finance_agent = Agent(
    role="Finance Expert",
    goal="Answer financial and payroll-related questions with clarity and correctness.",
    backstory=(
        "You are a Finance department specialist helping employees with "
        "salary disbursement, reimbursements, invoices, and tax-related FAQs."
    ),
    llm=llm,
)

it_agent = Agent(
    role="IT Support Specialist",
    goal="Provide answers and solutions to IT-related issues and technical queries.",
    backstory=(
        "You are an IT helpdesk assistant responsible for helping users with "
        "email setup, VPN access, password resets, and troubleshooting technical errors."
    ),
    llm=llm,
)


BOOST_KEYWORDS = {
    "salary": ["salary", "salaries", "pay", "paid", "credited", "credit"],
    "leave": ["leave", "leaves", "paid leave", "pto", "vacation"],
    "password": ["password", "reset", "forgot", "change password"],
}

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _keyword_boost(question: str, q_line: str) -> float:
    q_lower = question.lower()
    ql_lower = q_line.lower()
    boost = 0.0
    for _, words in BOOST_KEYWORDS.items():
        if any(w in q_lower for w in words) and any(w in ql_lower for w in words):
            boost += 0.2
    return boost



def get_answer(domain: str, question: str) -> str:
    domain = (domain or "").strip().lower()
    domain_map = {
        "hr": "data/hr_faq.txt",
        "finance": "data/finance_faq.txt",
        "it": "data/it_faq.txt"
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))
    relative = domain_map.get(domain, "")
    path = os.path.normpath(os.path.join(base_dir, "..", relative))

    print(f"[DEBUG] Domain detected: '{domain}'")
    print(f"[DEBUG] Looking for file: {path}")

    # Default fallback message
    fallback_message = f"Sorry, I couldn’t find a saved answer for your {domain.upper()} question. Let me help you with one."

    if not os.path.exists(path):
        print("[DEBUG] No data file found for this domain. Using AI fallback.")
        return _generate_fallback_answer(domain, question)

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    question_norm = question.lower().strip()
    best_idx = None
    best_score = 0.0

    for i, line in enumerate(lines):
        if line.lower().startswith("q:"):
            q_line = line.lower().replace("q:", "").strip()
            sim = _similarity(question_norm, q_line)
            sim += _keyword_boost(question_norm, q_line)

            if sim > best_score:
                best_score = sim
                best_idx = i

    THRESHOLD = 0.60

    if best_idx is not None and best_score >= THRESHOLD:
        if best_idx + 1 < len(lines):
            answer_line = lines[best_idx + 1].replace("A:", "").strip()
            print(f"[DEBUG] FAQ match found with score {best_score:.3f}")
            return answer_line


    print(f"[DEBUG] No strong FAQ match found (best_score={best_score:.3f}). Using LLM fallback.")
    return _generate_fallback_answer(domain, question)


def _generate_fallback_answer(domain: str, question: str) -> str:
    """Uses the correct domain LLM agent to generate an intelligent answer."""
    agent_map = {
        "hr": hr_agent,
        "finance": finance_agent,
        "it": it_agent
    }

    agent = agent_map.get(domain, hr_agent)
    prompt = f"The user asked: '{question}'. Please give a clear and professional answer related to {domain.upper()}."
    try:
        response = agent.run(prompt)
        return response or f"Sorry, I couldn't find a perfect answer, but this is my best attempt: {response}"
    except Exception as e:
        print(f"[ERROR] Fallback generation failed: {e}")
        return "Sorry, I couldn’t generate an answer right now."
