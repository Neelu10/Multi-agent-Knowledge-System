import streamlit as st
import os
from llm_setup import client
from agents.router_agent import classify_question
import difflib
from streamlit_extras.add_vertical_space import add_vertical_space

# ====== CONFIG ======
st.set_page_config(page_title="AI Company Assistant", layout="wide")

# Custom CSS for visual appeal
st.markdown("""
<style>
    /* Page background */
    .stApp {
        background: linear-gradient(135deg, #f4f7fa, #e8eff7);
        color: #222;
        font-family: 'Poppins', sans-serif;
    }
    /* Title */
    .title-container {
        text-align: center;
        padding: 20px 0;
        border-radius: 16px;
    }
    .title {
        font-size: 2.2rem;
        font-weight: 600;
        color: #003366;
    }
    .subtitle {
        color: #555;
        font-size: 1rem;
    }
    /* Input box */
    .stTextInput>div>div>input {
        border: 2px solid #004aad !important;
        border-radius: 12px !important;
        padding: 10px;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #004aad, #0073e6);
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.4em;
        border: none;
        transition: 0.3s ease;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0055cc, #0088ff);
        transform: scale(1.03);
    }
    /* Answer box */
    .answer-card {
        background-color: white;
        padding: 1.2em 1.5em;
        border-radius: 15px;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
        border-left: 6px solid #004aad;
        margin-top: 20px;
        animation: fadeIn 0.6s ease;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
</style>
""", unsafe_allow_html=True)

# ====== HEADER ======
st.markdown("""
<div class='title-container'>
    <div class='title'>💼 AI-Powered Company Assistant</div>
    <div class='subtitle'>Get instant help with HR, Finance, and IT-related queries</div>
</div>
""", unsafe_allow_html=True)

# ====== FILE LOADING ======
def load_text_files():
    data = {}
    for domain in ["HR", "Finance", "IT"]:
        filename = f"{domain.lower()}.txt"
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                data[domain] = f.read().split("\n")
        else:
            data[domain] = []
    return data

faq_data = load_text_files()

# ====== SEARCH FUNCTION ======
def find_best_match(question, domain_data):
    """Find closest question in the text file using fuzzy matching."""
    if not domain_data:
        return None
    matches = difflib.get_close_matches(question, domain_data, n=1, cutoff=0.6)
    return matches[0] if matches else None

# ====== MAIN FUNCTION ======
def get_answer(question):
    # Step 1: Classify domain
    domain = classify_question(question)
    if domain not in ["HR", "Finance", "IT"]:
        domain = "HR"  # fallback

    # Step 2: Try to find in text file
    matched_line = find_best_match(question, faq_data[domain])

    if matched_line:
        # Step 3: Ask LLM to rephrase the answer nicely
        prompt = f"Improve this FAQ answer for clarity and tone:\nQuestion: {question}\nAnswer: {matched_line}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        improved_answer = response.choices[0].message.content.strip()
        return domain, improved_answer

    # Step 4: If not found, generate fresh answer
    else:
        fallback_prompt = f"This question belongs to {domain}. Please give a helpful and accurate answer:\n\n{question}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": fallback_prompt}],
        )
        return domain, response.choices[0].message.content.strip()

# ====== UI ======
add_vertical_space(1)
question = st.text_input("💬 Ask your question here (e.g., 'When is salary credited?')")

if st.button("✨ Get Answer"):
    if question.strip():
        with st.spinner("🤖 Thinking..."):
            domain, answer = get_answer(question)
            st.markdown(
                f"""
                <div class="answer-card">
                    <h4>📂 Category: {domain}</h4>
                    <p style="font-size:1.05rem;">{answer}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ Please enter a question to proceed.")

# Footer
st.markdown("""
<hr style="margin-top:40px;">
<div style='text-align:center; color: #777; font-size: 0.9rem;'>
    Built using <b>CrewAI</b> and <b>OpenAI</b> | Smart FAQ Assistant for Enterprises
</div>
""", unsafe_allow_html=True)
