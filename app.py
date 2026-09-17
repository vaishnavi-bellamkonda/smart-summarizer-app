import streamlit as st
import PyPDF2
from transformers import pipeline
import openai
import os

# Page Configuration
st.set_page_config(page_title="Smart Doc Summarizer", page_icon="📝", layout="wide")

st.title("📝 Advanced Smart Text & Document Summarizer")
st.caption("Custom role-based summaries, action item extraction, and hybrid local/cloud execution.")

# Sidebar Settings
st.sidebar.header("⚙️ Configuration")

# Engine Selection
execution_mode = st.sidebar.radio(
    "Choose Summarization Engine:",
    ["OpenAI (Cloud - Fast & Advanced)", "HuggingFace BART (Local - Private & Free)"]
)

# Role / Persona Selection
target_persona = st.sidebar.selectbox(
    "Target Persona Output:",
    ["General Summary", "Executive (Action Items & ROI)", "Technical (Specs & Code)", "Student (Key Concepts & Quiz Prep)"]
)

# Output Format Selection
summary_length = st.sidebar.select_slider(
    "Summary Detail Level:",
    options=["Brief (Bullet Points)", "Medium (Structured Sections)", "Detailed (In-depth Analysis)"]
)

api_key = ""
if "OpenAI" in execution_mode:
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text

def chunk_text(text, chunk_size=1000):
    """Simple built-in text chunker replacing langchain dependency."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def summarize_with_openai(text, persona, length, key):
    client = openai.OpenAI(api_key=key)
    prompt = f"""
    You are an expert document summarizer. Summarize the following document according to these criteria:
    - Persona: {persona}
    - Detail Level: {length}
    
    Structure the response with clear headings, core key takeaways, and an 'Actionable Items' section if applicable.
    
    Document Text:
    {text[:12000]} 
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

@st.cache_resource
def load_local_model():
    # Downloads BART summarization pipeline locally
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

def summarize_with_local(text, length):
    summarizer = load_local_model()
    docs = chunk_text(text, chunk_size=1000)
    
    max_len = 150 if "Detailed" in length else 75
    min_len = 50 if "Detailed" in length else 25
    
    summary = summarizer(docs[0], max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']

# --- Main App Interface ---

tab1, tab2 = st.tabs(["📄 Document Upload", "✏️ Paste Raw Text"])
input_text = ""

with tab1:
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            input_text = extract_text_from_pdf(uploaded_file)
        else:
            input_text = str(uploaded_file.read().decode("utf-8"))
        st.success(f"File uploaded successfully! Extracted ~{len(input_text.split())} words.")

with tab2:
    pasted_text = st.text_area("Paste long article text here...", height=250)
    if pasted_text:
        input_text = pasted_text

# Execution Trigger
if st.button("🚀 Generate Specialized Summary"):
    if not input_text.strip():
        st.warning("Please upload a file or paste text first.")
    else:
        with st.spinner("Processing document..."):
            try:
                if "OpenAI" in execution_mode:
                    if not api_key:
                        st.error("Please enter a valid OpenAI API key in the sidebar.")
                    else:
                        result = summarize_with_openai(input_text, target_persona, summary_length, api_key)
                        
                        st.markdown("### 📊 Generated Summary")
                        st.markdown(result)
                        
                        st.download_button(
                            label="📥 Download Summary as Markdown",
                            data=result,
                            file_name="summary.md",
                            mime="text/markdown"
                        )
                else:
                    result = summarize_with_local(input_text, summary_length)
                    st.markdown("### 📊 Local BART Model Summary")
                    st.info("Note: Persona tuning is limited under the free offline transformer pipeline.")
                    st.write(result)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
