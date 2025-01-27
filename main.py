import streamlit as st
import PyPDF2
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import google.generativeai as genai
import json
import re
import random
import os

load_dotenv()
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
#############################################
# 1. Custom CSS Styling
#############################################
# We can insert some CSS for fonts, layout, etc.

page_style = page_style = """
<style>
/* Make the main container a bit wider */
.main .block-container {
    max-width: 800px;
}

/* Set background color */
body {
    background-color: #121212; /* Darker background for contrast */
}

/* Custom heading styles */
h1 {
    color: #008CFF;  /* Brighter, more vibrant blue */
    font-weight: 900;
    text-align: center;
    font-size: 2.2rem; /* Make it stand out */
    margin-bottom: 0.5em;
}

/* Subheading styles */
h2, h3 {
    color: #FFC107;  /* Light golden yellow */
    font-size: 1.5rem;
    font-weight: bold;
}

/* Subtext / Descriptions */
p, .description-text {
    color: #D1D1D1; /* Light grey for better visibility */
    font-size: 1.1rem;
    text-align: center;
    letter-spacing: 0.5px; /* Improves readability */
    margin-bottom: 1em;
}

/* Buttons and messages */
.stButton button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 0.6em 1em;
    text-decoration: none;
    font-size: 1rem;
    margin: 0.5em 0.25em;
    cursor: pointer;
    border-radius: 5px;
    transition: background-color 0.3s ease-in-out;
}

.stButton button:hover {
    background-color: #45a049;
}

/* Quiz container styling */
.quiz-container {
    background-color: #1E1E1E; /* Slightly darker gray for contrast */
    padding: 1.5em;
    border-radius: 8px;
    margin-bottom: 1.5em;
    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    border-left: 5px solid #FFC107; /* Gold accent */
}

/* Score styling */
.score-text {
    font-weight: bold;
    font-size: 1.4rem;
    color: #FFC107;
    text-align: center;
    padding: 0.5em 0;
}
</style>
"""



#############################################
# 2. Utility Functions (Backend)
#############################################

def extract_text_from_pdf(file_obj):
    """
    Extracts text from a single PDF file object using PyPDF2.
    Returns a string of all extracted text.
    """
    text = ""
    pdf = PyPDF2.PdfReader(file_obj)
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_key_sentences(full_text, sentence_count=12):
    """
    Uses Sumy (LexRank) to extract the top `sentence_count` sentences 
    as an 'extractive summary' from the text.
    """
    parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    extracted_sents = [str(s) for s in summary_sentences]
    return extracted_sents

def strict_quiz_prompt(extracted_sents, difficulty="medium"):
    """
    Builds a strict prompt that demands valid JSON only, 
    for a 10-question quiz with 4 options each.
    """
    context_text = "\n".join(extracted_sents)

    prompt = f"""
Below are some key sentences extracted from class notes (difficulty: {difficulty}):

{context_text}

PLEASE FOLLOW THESE RULES:
1. Return only valid JSON, and nothing else.
2. Do not include code fences, markdown, or extra explanation.
3. The JSON must have this structure:
{{
  "quiz": [
    {{
      "question": "...",
      "options": ["...","...","...","..."],
      "answer_index": 0
    }},
    ...
  ]
}}

Now create exactly 10 multiple-choice questions based on these sentences, 
each with 4 options, plus the correct answer_index. 
Return only valid JSON.
"""
    return prompt

def generate_quiz_with_gemini(prompt_text):
    """
    Calls Gemini with the given prompt_text and returns raw LLM response as a string.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt_text)
    return response.text  # raw string

def extract_json_from_text(text):
    """
    Attempt to isolate valid JSON by capturing the first '{' through the final '}'.
    This helps remove disclaimers or other extraneous text.
    """
    text = text.strip()
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def parse_quiz_json(json_str):
    """
    Parses the quiz JSON string returned by Gemini.
    Returns a Python dict or None if parse fails.
    """
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError:
        return None

#############################################
# 3. Streamlit UI
#############################################

def main():
    # Inject our custom CSS
    st.markdown(page_style, unsafe_allow_html=True)

    # Optional: You can add a banner/logo image
    # st.image("https://your-domain.com/banner.png", use_column_width=True)

    st.title("PDF Quiz Generator")

    st.write(
        """
        <p style='text-align: font-weight: 900 ;center; font-size:1.1rem; color:#333;'>
        Upload one or more PDF files and generate a 10-question quiz.
        </p>
        """,
        unsafe_allow_html=True
    )

    # 1) Configure Gemini with your API key
    # (Replace with your real key or use st.secrets in production)
    genai.configure(api_key=GOOGLE_API_KEY)

    # Step A: File Upload (center this with columns)
    col1, col2, col3 = st.columns([1,4,1])
    with col2:
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

    # Let user select quiz difficulty
    st.subheader("Quiz Difficulty")
    difficulty = st.radio(
        "Choose difficulty level:",
        options=["Easy", "Medium", "Hard"],
        index = None,
        horizontal=True
    )

    if uploaded_files:
        if st.button("Generate Quiz"):
            # 2) Combine text from all uploaded PDFs
            combined_text = ""
            for f in uploaded_files:
                pdf_text = extract_text_from_pdf(f)
                combined_text += pdf_text + "\n"

            # 3) Extract ~12 key sentences
            key_sents = extract_key_sentences(combined_text, sentence_count=12)

            # 4) Build a strict prompt to produce valid JSON
            prompt = strict_quiz_prompt(key_sents, difficulty=difficulty)

            # 5) Call Gemini
            quiz_json_raw = generate_quiz_with_gemini(prompt)

            # 6) Post-process to isolate JSON
            quiz_json_str = extract_json_from_text(quiz_json_raw)

            # 7) Parse JSON
            quiz_data = parse_quiz_json(quiz_json_str)
            if quiz_data is None:
                st.error("Failed to parse quiz JSON from Gemini. Please try again.")
            else:
                st.session_state["quiz_data"] = quiz_data
                st.success("Quiz generated! Scroll down to attempt it.")

    # Step B: Display the quiz if quiz_data is available
    if "quiz_data" in st.session_state and st.session_state["quiz_data"] is not None:
        quiz_data = st.session_state["quiz_data"]
        quiz_items = quiz_data.get("quiz", [])
        
        if quiz_items:
            st.markdown("---")

            # We'll store user answers in session_state
            if "user_answers" not in st.session_state:
                st.session_state["user_answers"] = [None] * len(quiz_items)

            # Put the quiz inside a container
            with st.container():
                for i, item in enumerate(quiz_items):
                    st.markdown(f"<div class='quiz-container'>", unsafe_allow_html=True)
                    st.write(f"**Q{i+1}. {item['question']}**")
                    options = item["options"]
                    

                    default_idx = 0 if st.session_state["user_answers"][i] is None else st.session_state["user_answers"][i]
                    
                    # Let user pick radio button
                    answer = st.radio(
                        label=f"",
                        options=options,
                        index=default_idx,
                        key=f"question_{i}"
                    )
                    selected_index = options.index(answer)
                    st.session_state["user_answers"][i] = selected_index
                    st.markdown("</div>", unsafe_allow_html=True)

            # Submit answers
            if st.button("Submit Answers"):
                score = 0
                for i, item in enumerate(quiz_items):
                    correct_idx = item["answer_index"]
                    user_idx = st.session_state["user_answers"][i]
                    if user_idx == correct_idx:
                        score += 1

                st.markdown(f"<p class='score-text'>Your Score: {score}/{len(quiz_items)}</p>", unsafe_allow_html=True)

                # Reveal correct answers
                with st.expander("See Correct Answers"):
                    for i, item in enumerate(quiz_items):
                        correct_idx = item["answer_index"]
                        st.write(f"Q{i+1} Answer: {item['options'][correct_idx]}")
        else:
            st.info("No quiz questions found. Please generate a quiz first.")


if __name__ == "__main__":
    main()
