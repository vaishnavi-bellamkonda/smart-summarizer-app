import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import PyPDF2
import re
import html


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="Smart Text & Document Summarizer",
    page_icon="📝",
    layout="wide"
)


# ==========================================================
# CUSTOM DESIGN
# ==========================================================

st.markdown(
    """
    <style>

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777777;
        margin-bottom: 30px;
    }

    .summary-box {
        background-color: #ffffff;
        color: #111111;
        border: 2px solid #cccccc;
        border-radius: 12px;
        padding: 25px;
        font-size: 18px;
        line-height: 1.8;
        min-height: 150px;
        margin-top: 10px;
    }

    .summary-box p {
        color: #111111 !important;
    }

    .summary-box span {
        color: #111111 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    '<div class="title">Smart Text & Document Summarizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An AI-powered application for summarizing text and documents'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# MODEL
# ==========================================================

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


@st.cache_resource
def load_model():

    st.info("Loading AI summarization model for the first time...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    return tokenizer, model, device


# ==========================================================
# CLEAN TEXT
# ==========================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================================
# PDF READER
# ==========================================================

def read_pdf(file):

    try:

        reader = PyPDF2.PdfReader(file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return clean_text(text)

    except Exception as e:

        st.error(
            f"Error reading PDF: {e}"
        )

        return ""


# ==========================================================
# TXT READER
# ==========================================================

def read_txt(file):

    try:

        data = file.read()

        try:
            text = data.decode("utf-8")

        except UnicodeDecodeError:

            text = data.decode("latin-1")

        return clean_text(text)

    except Exception as e:

        st.error(
            f"Error reading TXT file: {e}"
        )

        return ""


# ==========================================================
# TEXT CHUNKING
# ==========================================================

def create_chunks(text, words_per_chunk=300):

    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        words_per_chunk
    ):

        chunk = " ".join(
            words[i:i + words_per_chunk]
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ==========================================================
# SUMMARIZE ONE CHUNK
# ==========================================================

def summarize_chunk(
    text,
    tokenizer,
    model,
    device,
    summary_type
):

    if summary_type == "Short":

        minimum_length = 20
        maximum_length = 80

    else:

        minimum_length = 35
        maximum_length = 130


    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )


    input_ids = encoded[
        "input_ids"
    ].to(device)

    attention_mask = encoded[
        "attention_mask"
    ].to(device)


    with torch.no_grad():

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=maximum_length,
            min_length=minimum_length,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )


    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return result.strip()


# ==========================================================
# COMPLETE SUMMARIZATION
# ==========================================================

def create_summary(text, summary_type):

    tokenizer, model, device = load_model()

    chunks = create_chunks(
        text,
        words_per_chunk=300
    )

    summaries = []

    progress = st.progress(0)

    total_chunks = len(chunks)

    for number, chunk in enumerate(chunks):

        result = summarize_chunk(
            chunk,
            tokenizer,
            model,
            device,
            summary_type
        )

        if result:

            summaries.append(result)

        progress.progress(
            (number + 1) / total_chunks
        )

    progress.empty()

    return " ".join(summaries)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("Settings")

    summary_type = st.radio(
        "Summary Type",
        [
            "Short",
            "Detailed"
        ]
    )

    st.markdown("---")

    st.subheader("Supported Files")

    st.write(
        "PDF\n\nTXT\n\nPasted Text"
    )

    st.markdown("---")

    st.info(
        "Uses a transformer-based NLP model "
        "for automatic summarization."
    )


# ==========================================================
# TEXT INPUT
# ==========================================================

st.subheader("Enter Text")

text_input = st.text_area(
    "Paste your text below:",
    height=250,
    placeholder="Paste an article, report, notes or any long text here..."
)


# ==========================================================
# FILE UPLOAD
# ==========================================================

st.subheader("Upload Document")

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=[
        "pdf",
        "txt"
    ]
)


# ==========================================================
# GET DOCUMENT TEXT
# ==========================================================

document_text = ""


if uploaded_file is not None:

    if uploaded_file.name.lower().endswith(".pdf"):

        document_text = read_pdf(
            uploaded_file
        )

    elif uploaded_file.name.lower().endswith(".txt"):

        document_text = read_txt(
            uploaded_file
        )


# ==========================================================
# DOCUMENT INFORMATION
# ==========================================================

if document_text:

    st.success(
        f"Successfully loaded: {uploaded_file.name}"
    )

    st.info(
        f"Document contains approximately "
        f"{len(document_text.split())} words."
    )


# ==========================================================
# BUTTON
# ==========================================================

st.markdown("---")

generate = st.button(
    "Generate Summary",
    type="primary",
    use_container_width=True
)


# ==========================================================
# GENERATE SUMMARY
# ==========================================================

if generate:

    # Choose document or pasted text

    if document_text:

        text = document_text

    elif text_input.strip():

        text = clean_text(
            text_input
        )

    else:

        st.warning(
            "Please paste some text or upload a document."
        )

        st.stop()


    # Check length

    word_count = len(
        text.split()
    )

    if word_count < 30:

        st.warning(
            "Please provide at least 30 words."
        )

        st.stop()


    # Limit extremely large documents

    if word_count > 5000:

        st.warning(
            "The document is larger than 5000 words. "
            "Only the first 5000 words will be processed."
        )

        text = " ".join(
            text.split()[:5000]
        )


    # Generate

    with st.spinner(
        "AI is generating your summary..."
    ):

        try:

            summary = create_summary(
                text,
                summary_type
            )

        except Exception as e:

            st.error(
                "Something went wrong while generating the summary."
            )

            st.exception(e)

            st.stop()


    # ======================================================
    # SHOW SUMMARY
    # ======================================================

    if summary:

        st.subheader(
            f"{summary_type} Summary"
        )

        # Escape generated text safely

        safe_summary = html.escape(
            summary
        )

        st.markdown(
            f"""
            <div class="summary-box">
                {safe_summary}
            </div>
            """,
            unsafe_allow_html=True
        )


        # ==================================================
        # STATISTICS
        # ==================================================

        original_words = len(
            text.split()
        )

        summary_words = len(
            summary.split()
        )

        reduction = (
            1 -
            summary_words / original_words
        ) * 100


        st.markdown("---")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Original Words",
                original_words
            )


        with col2:

            st.metric(
                "Summary Words",
                summary_words
            )


        with col3:

            st.metric(
                "Reduction",
                f"{reduction:.1f}%"
            )


        # ==================================================
        # DOWNLOAD
        # ==================================================

        st.download_button(
            "Download Summary",
            data=summary,
            file_name="smart_summary.txt",
            mime="text/plain",
            use_container_width=True
        )


    else:

        st.error(
            "No summary was generated. Please try again."
        )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#888888;">
        Smart Text & Document Summarizer
        <br>
        NLP • Transformers • Python • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
