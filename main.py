# app.py
import streamlit as st
import PyPDF2
import docx
import pandas as pd
import re
import io
import time
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy

# Optimize spaCy pipeline for keyword extraction
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Inject global custom CSS
st.markdown(
    """
    <style>
    body {
        font-family: "Roboto", sans-serif;
    }
    .st-emotion-cache-mtjnbi {
        width: 100%;
        margin: 0 auto;
        padding: 20px;
        padding-top: 50px;
        max-width: 1050px;
        background-color: #0e1117;
        color: white;
        border-radius: 10px;
    }
    [class^="css-"][class*="Sidebar"] {
        padding: 20px;
        background-color: #1d212d;
    }
    .st-emotion-cache-mtjnbi h1 {
        font-size: 50px;
        font-family: "Times New Roman", Times, serif;
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 300% 300%;
        -webkit-text-fill-color: transparent; 
        -webkit-background-clip: text; 
        animation: gradient-animation 10s ease infinite; 
    }
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .st-emotion-cache-1espb9k h1{
            background: linear-gradient(to right, #f32170, 
                    #ff6b08, #cf23cf, #eedd44); 
            -webkit-text-fill-color: transparent; 
            -webkit-background-clip: text; 
        }
        
        .st-emotion-cache-ah6jdd{
        font-size: 20px;
        opacity: 0.8;
        color:#ffffff8f;
        }
        
    
    .centered-table {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    .centered-table table {
        width: 80%;
        background: #1e2130;
        color: white;
        border: 1px solid #c5d5ef;
        border-collapse: collapse;
        border-radius: 10px;
        overflow: hidden;
    }
    .centered-table th, .centered-table td {
        padding: 10px;
        text-align: left;
        border: 1px solid #c5d5ef;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Setting the app title
st.title("AI-powered Resume Screening System")
st.text("Is your resume good enough?")
st.text("A free and fast AI resume checker doing crucial checks to ensure your resume is ready for interviews.")


# extracting text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# extracting text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])


# Remove unwanted characters/whitespaces
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s.,]", "", text)
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()


# Extract keywords using spaCy
def extract_keywords(text):
    doc = nlp(text)
    return list(set(token.text.lower() for token in doc if token.is_alpha and not token.is_stop))


# Compute similarity scores
def compute_similarity(resume_texts, job_description):
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    embeddings = model.encode([job_description] + resume_texts)
    return cosine_similarity([embeddings[0]], embeddings[1:])[0]


# Sidebar input for job description
st.sidebar.title("Job Description")
job_description = st.sidebar.text_area("Enter or paste Job Description below:",
                                       help="Paste the job description you want to match against.")

# Sidebar for resume upload
st.sidebar.title("Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload multiple resumes", type=["pdf", "docx"], accept_multiple_files=True)

similarity_weight = st.sidebar.slider("Weight for Similarity Score (%)", 0, 100, 70,
                                      help="Adjust slider to change the importance of Similarity Score.")
match_weight = 100 - similarity_weight  # Ensure weights sum to 100

rank_resumes = st.sidebar.button("Rank Resumes")

if rank_resumes:
    if not uploaded_files or not job_description.strip():
        st.error("Please upload resumes and provide the Job Description.")
    else:
        resume_texts = []
        progress_bar = st.progress(0)  # Progress indicator

        job_description = preprocess_text(job_description)
        jd_keywords = extract_keywords(job_description)

        for idx, uploaded_file in enumerate(uploaded_files):
            if uploaded_file.name.endswith(".pdf"):
                text = preprocess_text(extract_text_from_pdf(uploaded_file))
            elif uploaded_file.name.endswith(".docx"):
                text = preprocess_text(extract_text_from_docx(uploaded_file))
            else:
                st.error(f"Unsupported file format: {uploaded_file.name}")
                continue

            max_characters = 5000
            if len(text) > max_characters:
                text = text[:max_characters]
                st.warning(f"Truncated {uploaded_file.name} to {max_characters} characters.")

            resume_texts.append(text)

            progress_bar.progress((idx + 1) / len(uploaded_files))

        # Calculate Similarity Scores
        similarity_scores = compute_similarity(resume_texts, job_description)

        # Keyword Matches
        keyword_matches = []
        for resume_text in resume_texts:
            resume_keywords = extract_keywords(resume_text)
            match_count = len(set(jd_keywords) & set(resume_keywords))
            keyword_matches.append(match_count)

        # Create Rankings DataFrame
        rankings = pd.DataFrame({
            "Resume": [uploaded_file.name for uploaded_file in uploaded_files],
            "Similarity Score": similarity_scores,
            "Keyword Matches": keyword_matches,
        })

        # Calculate Final Score
        rankings["Final Score"] = (rankings["Similarity Score"] * (similarity_weight / 100) +
                                   rankings["Keyword Matches"] * (match_weight / 100))
        rankings = rankings.sort_values(by="Final Score", ascending=False)

        st.success("Resumes Ranked Successfully!")

        # Display Rankings DataFrame
        st.markdown('<div class="centered-table">', unsafe_allow_html=True)
        st.table(rankings)
        st.markdown('</div>', unsafe_allow_html=True)

        # Download Rankings as Excel
        output = io.BytesIO()
        rankings.to_excel(output, index=False)
        output.seek(0)

        st.download_button(
            label="Download Rankings as Excel",
            data=output,
            file_name="resume_rankings.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # Visualize Rankings with Altair
        st.altair_chart(
            alt.Chart(rankings).mark_bar().encode(
                x="Resume",
                y="Final Score",
                color="Keyword Matches",
                tooltip=["Resume", "Final Score", "Similarity Score", "Keyword Matches"],
            ).properties(title="Resume Rankings")
        )
