import streamlit as st
import pypdf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Smart Text & Document Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Advanced Smart Text & Document Summarizer")
st.caption("Custom role-based summaries, action item extraction, and hybrid local/cloud execution.")

# ==================== SIDEBAR CONFIGURATION ====================
st.sidebar.header("⚙️ Configuration")

engine = st.sidebar.radio(
    "Choose Summarization Engine:",
    ["OpenAI (Cloud - Fast & Advanced)", "HuggingFace BART (Local - Private & Free)"]
)

persona = st.sidebar.selectbox(
    "Target Persona Output:",
    ["General Summary", "Executive Brief", "Action Items & Next Steps", "Technical Overview"]
)

detail_level = st.sidebar.select_slider(
    "Summary Detail Level:",
    options=["Brief (Bullet Points)", "Balanced", "Detailed Report"]
)

openai_api_key = ""
if "OpenAI" in engine:
    openai_api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")

# ==================== HELPER FUNCTIONS ====================

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF document."""
    pdf_reader = pypdf.PdfReader(pdf_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text


@st.cache_resource
def load_hf_model():
    """Load and cache Hugging Face BART model and tokenizer locally."""
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def summarize_with_hf(text, detail_level):
    """Summarize text locally using Hugging Face BART without relying on pipeline task names."""
    tokenizer, model = load_hf_model()

    # Determine max/min length based on detail level
    max_length_map = {
        "Brief (Bullet Points)": 80,
        "Balanced": 150,
        "Detailed Report": 300
    }
    max_len = max_length_map.get(detail_level, 150)
    min_len = int(max_len * 0.3)

    # Tokenize input text (chunking to 1024 tokens max for BART)
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generate summary tokens
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def summarize_with_openai(text, api_key, persona, detail_level):
    """Summarize text using OpenAI API."""
    import openai
    client = openai.OpenAI(api_key=api_key)

    prompt = f"""
    You are an expert assistant. Summarize the following document.
    Target Audience/Persona: {persona}
    Detail Level: {detail_level}

    Document Text:
    {text[:12000]}  # Truncated to fit token limits
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional text summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


# ==================== MAIN UI & INPUTS ====================

tab1, tab2 = st.tabs(["📄 Document Upload", "✏️ Paste Raw Text"])

text_content = ""

with tab1:
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text_content = extract_text_from_pdf(uploaded_file)
        else:
            text_content = uploaded_file.read().decode("utf-8")

        word_count = len(text_content.split())
        st.success(f"File uploaded successfully! Extracted ~{word_count} words.")

with tab2:
    pasted_text = st.text_area("Paste text here for summarization:", height=250)
    if pasted_text:
        text_content = pasted_text

# ==================== SUMMARIZE ACTION ====================

if st.button("🚀 Generate Specialized Summary", type="primary"):
    if not text_content.strip():
        st.error("Please upload a file or paste text first.")
    elif "OpenAI" in engine and not openai_api_key.strip():
        st.error("Please enter a valid OpenAI API key in the sidebar.")
    else:
        with st.spinner("Generating summary... Please wait."):
            try:
                if "OpenAI" in engine:
                    summary = summarize_with_openai(text_content, openai_api_key, persona, detail_level)
                else:
                    summary = summarize_with_hf(text_content, detail_level)

                st.subheader("📌 Generated Summary")
                st.markdown(summary)

            except Exception as e:
                st.error(f"An error occurred during summarization: {str(e)}")
