import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import PyPDF2
import re
import html


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Text & Document Summarizer",
    page_icon="📝",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777777;
        margin-bottom: 30px;
    }

    .summary-box {
        background-color: #ffffff !important;
        color: #111111 !important;
        border: 2px solid #cccccc;
        border-radius: 12px;
        padding: 25px;
        font-size: 18px;
        line-height: 1.8;
        min-height: 150px;
        overflow-wrap: break-word;
    }

    .summary-box * {
        color: #111111 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

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


# =========================================================
# MODEL
# =========================================================

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    return tokenizer, model, device


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# READ PDF
# =========================================================

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


# =========================================================
# READ TXT
# =========================================================

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


# =========================================================
# SPLIT LONG TEXT
# =========================================================

def create_chunks(
    text,
    words_per_chunk=300
):

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


# =========================================================
# SUMMARIZE ONE CHUNK
# =========================================================

def summarize_chunk(
    text,
    tokenizer,
    model,
    device,
    summary_type
):

    if summary_type == "Short":

        min_length = 20
        max_length = 80

    else:

        min_length = 35
        max_length = 130


    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )


    input_ids = inputs[
        "input_ids"
    ].to(device)

    attention_mask = inputs[
        "attention_mask"
    ].to(device)


    with torch.no_grad():

        summary_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )


    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary.strip()


# =========================================================
# GENERATE SUMMARY
# =========================================================

def generate_summary(
    text,
    summary_type
):

    tokenizer, model, device = load_model()

    chunks = create_chunks(
        text,
        300
    )

    summaries = []

    progress = st.progress(0)

    total = len(chunks)

    for i, chunk in enumerate(chunks):

        summary = summarize_chunk(
            chunk,
            tokenizer,
            model,
            device,
            summary_type
        )

        if summary:
            summaries.append(summary)

        progress.progress(
            (i + 1) / total
        )

    progress.empty()

    return " ".join(summaries)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Summarizer Settings")

    summary_type = st.radio(
        "Choose Summary Type",
        [
            "Short",
            "Detailed"
        ]
    )

    st.markdown("---")

    st.subheader("Supported Inputs")

    st.write(
        "• Pasted Text\n\n"
        "• PDF Documents\n\n"
        "• TXT Files"
    )

    st.markdown("---")

    st.info(
        "Powered by Python, Streamlit "
        "and Transformer-based NLP."
    )


# =========================================================
# TEXT INPUT
# =========================================================

st.subheader("Enter Your Text")

text_input = st.text_area(
    "Paste your text below:",
    height=250,
    placeholder="Paste an article, report or any long text here..."
)


# =========================================================
# FILE UPLOAD
# =========================================================

st.subheader("Upload Document")

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=[
        "pdf",
        "txt"
    ]
)


# =========================================================
# EXTRACT DOCUMENT
# =========================================================

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


# =========================================================
# DOCUMENT INFORMATION
# =========================================================

if document_text:

    st.success(
        f"Successfully loaded: {uploaded_file.name}"
    )

    st.info(
        f"Document contains approximately "
        f"{len(document_text.split())} words."
    )


# =========================================================
# GENERATE BUTTON
# =========================================================

st.markdown("---")

generate = st.button(
    "Generate Summary",
    type="primary",
    use_container_width=True
)


# =========================================================
# PROCESS SUMMARY
# =========================================================

if generate:

    if document_text:

        text = document_text

    elif text_input.strip():

        text = clean_text(
            text_input
        )

    else:

        st.warning(
            "Please paste text or upload a document."
        )

        st.stop()


    word_count = len(
        text.split()
    )


    if word_count < 30:

        st.warning(
            "Please provide at least 30 words."
        )

        st.stop()


    if word_count > 5000:

        st.warning(
            "The document is larger than 5000 words. "
            "Only the first 5000 words will be processed."
        )

        text = " ".join(
            text.split()[:5000]
        )


    # =====================================================
    # GENERATE
    # =====================================================

    with st.spinner(
        "AI is generating your summary..."
    ):

        try:

            summary = generate_summary(
                text,
                summary_type
            )

        except Exception as e:

            st.error(
                "An error occurred while generating "
                "the summary."
            )

            st.exception(e)

            st.stop()


    # =====================================================
    # DISPLAY SUMMARY
    # =====================================================

    if summary:

        st.subheader(
            f"{summary_type} Summary"
        )

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


        # =================================================
        # STATISTICS
        # =================================================

        original_words = len(
            text.split()
        )

        summary_words = len(
            summary.split()
        )

        reduction = (
            1 -
            (
                summary_words /
                original_words
            )
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


        # =================================================
        # DOWNLOAD
        # =================================================

        st.download_button(
            "Download Summary",
            data=summary,
            file_name="smart_summary.txt",
            mime="text/plain",
            use_container_width=True
        )


    else:

        st.error(
            "No summary was generated."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#888888;
        padding:15px;
    ">
        Smart Text & Document Summarizer
        <br>
        NLP • Transformers • Python • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
