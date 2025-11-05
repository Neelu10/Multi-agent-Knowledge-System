from agents.router_agent import classify_question
from agents.domain_agent import get_answer, hr_agent, finance_agent, it_agent
from agents.explain_agent import simplify_answer
from llm_setup import client
import json

def get_faq_response(question: str):
    """
    Handles routing, domain retrieval, LLM fallback, and answer refinement.
    Returns a step dictionary for Streamlit UI visualization.
    """

    # Step 1️⃣ – Classify the question into a domain
    domain = classify_question(question)
    steps = {"router": f"🔍 Routed to domain: {domain}"}

    # Step 2️⃣ – Try fetching from local FAQ
    raw_answer = get_answer(domain, question)
    steps["domain"] = f"📚 FAQ Lookup: {raw_answer}"

    # Step 3️⃣ – If not found, query a domain-specific LLM agent
    if not raw_answer or "not found" in raw_answer.lower():
        steps["domain"] += "\n\n❌ Not found in text files — using LLM reasoning."

        prompt = (
            f"You are an expert in {domain} department.\n"
            f"Answer this employee's question accurately and professionally:\n\n"
            f"Question: {question}"
        )

        try:
            if domain.lower() == "hr":
                raw_answer = hr_agent.run(prompt)
            elif domain.lower() == "finance":
                raw_answer = finance_agent.run(prompt)
            elif domain.lower() == "it":
                raw_answer = it_agent.run(prompt)
            else:
                # Fallback to OpenAI directly
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                raw_answer = response.choices[0].message.content.strip()

        except Exception as e:
            steps["domain"] += f"\n⚠️ Error during LLM generation: {e}"
            raw_answer = "Sorry, I'm unable to generate an answer right now."

    # Step 4️⃣ – Refine & Simplify the final response
    try:
        final_answer = simplify_answer(raw_answer)
        steps["explain"] = "✨ Simplified and rewritten for clarity."
    except Exception as e:
        final_answer = raw_answer
        steps["explain"] = f"⚠️ Simplification skipped due to error: {e}"

    return steps, final_answer
