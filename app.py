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
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #AAAAAA;
        margin-bottom: 30px;
    }

    /* SUMMARY BOX */
    .summary-box {
        padding: 25px;
        border-radius: 12px;
        border: 2px solid #555555;
        background-color: #FFFFFF !important;
        color: #111111 !important;
        line-height: 1.8;
        font-size: 17px;
        min-height: 150px;
        white-space: normal;
        overflow-wrap: break-word;
    }

    .summary-box p {
        color: #111111 !important;
    }

    .summary-box * {
        color: #111111 !important;
    }

    .section-title {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">Smart Text & Document Summarizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Summarize long text, PDF documents and TXT files using NLP'
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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
# PDF EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    try:

        reader = PyPDF2.PdfReader(
            uploaded_file
        )

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return clean_text(text)

    except Exception as error:

        st.error(
            f"Could not read PDF: {error}"
        )

        return ""


# =========================================================
# TXT EXTRACTION
# =========================================================

def extract_txt_text(uploaded_file):

    try:

        content = uploaded_file.read()

        try:

            text = content.decode("utf-8")

        except UnicodeDecodeError:

            text = content.decode(
                "latin-1"
            )

        return clean_text(text)

    except Exception as error:

        st.error(
            f"Could not read TXT file: {error}"
        )

        return ""


# =========================================================
# SPLIT TEXT INTO CHUNKS
# =========================================================

def split_text(text, max_words=350):

    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        max_words
    ):

        chunk = " ".join(
            words[i:i + max_words]
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
    min_length,
    max_length
):

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
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary.strip()


# =========================================================
# GENERATE COMPLETE SUMMARY
# =========================================================

def generate_summary(
    text,
    summary_type
):

    tokenizer, model, device = load_model()

    chunks = split_text(
        text,
        max_words=350
    )

    summaries = []

    progress = st.progress(0)

    total = len(chunks)

    for index, chunk in enumerate(chunks):

        if summary_type == "Short":

            summary = summarize_chunk(
                chunk,
                tokenizer,
                model,
                device,
                min_length=20,
                max_length=80
            )

        else:

            summary = summarize_chunk(
                chunk,
                tokenizer,
                model,
                device,
                min_length=40,
                max_length=140
            )

        if summary:
            summaries.append(summary)

        progress.progress(
            (index + 1) / total
        )

    progress.empty()

    final_summary = " ".join(
        summaries
    )

    return final_summary


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

    st.subheader(
        "Supported Inputs"
    )

    st.write(
        """
        • Pasted text

        • PDF documents

        • TXT files
        """
    )

    st.markdown("---")

    st.info(
        "Powered by Python, Streamlit "
        "and Transformer-based NLP."
    )


# =========================================================
# TEXT INPUT
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Enter Your Text'
    '</div>',
    unsafe_allow_html=True
)

text_input = st.text_area(
    "Paste your article, notes, report or other text:",
    height=250,
    placeholder="Paste your text here..."
)


# =========================================================
# FILE UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Or Upload a Document'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload PDF or TXT file",
    type=[
        "pdf",
        "txt"
    ]
)


# =========================================================
# EXTRACT FILE TEXT
# =========================================================

document_text = ""


if uploaded_file is not None:

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):

        document_text = extract_pdf_text(
            uploaded_file
        )

    elif filename.endswith(".txt"):

        document_text = extract_txt_text(
            uploaded_file
        )


# =========================================================
# SHOW FILE INFORMATION
# =========================================================

if document_text:

    st.success(
        f"Successfully extracted text from "
        f"{uploaded_file.name}"
    )

    word_count = len(
        document_text.split()
    )

    st.info(
        f"Extracted approximately "
        f"{word_count} words."
    )


# =========================================================
# GENERATE BUTTON
# =========================================================

st.markdown("---")

generate_button = st.button(
    "Generate Summary",
    type="primary",
    use_container_width=True
)


# =========================================================
# SUMMARY PROCESS
# =========================================================

if generate_button:

    # -----------------------------------------------------
    # SELECT INPUT
    # -----------------------------------------------------

    if document_text:

        final_text = document_text

    elif text_input.strip():

        final_text = clean_text(
            text_input
        )

    else:

        st.warning(
            "Please enter some text or "
            "upload a PDF/TXT file."
        )

        st.stop()


    # -----------------------------------------------------
    # WORD COUNT
    # -----------------------------------------------------

    word_list = final_text.split()

    if len(word_list) < 30:

        st.warning(
            "Please provide at least "
            "30 words for summarization."
        )

        st.stop()


    # -----------------------------------------------------
    # LIMIT VERY LARGE DOCUMENTS
    # -----------------------------------------------------

    if len(word_list) > 5000:

        st.warning(
            "The document contains more than "
            "5000 words. Only the first 5000 "
            "words will be summarized."
        )

        final_text = " ".join(
            word_list[:5000]
        )


    # -----------------------------------------------------
    # GENERATE SUMMARY
    # -----------------------------------------------------

    with st.spinner(
        "Generating your summary..."
    ):

        try:

            summary = generate_summary(
                final_text,
                summary_type
            )

        except Exception as error:

            st.error(
                "An error occurred while "
                "generating the summary."
            )

            st.exception(error)

            st.stop()


    # -----------------------------------------------------
    # CHECK SUMMARY
    # -----------------------------------------------------

    if not summary:

        st.error(
            "The model did not generate a summary. "
            "Please try again with different text."
        )

        st.stop()


    # =====================================================
    # DISPLAY SUMMARY
    # =====================================================

    st.success(
        "Summary generated successfully!"
    )

    st.markdown(
        f'<div class="section-title">'
        f'{summary_type} Summary'
        f'</div>',
        unsafe_allow_html=True
    )


    # Escape HTML so generated text cannot break
    # the summary box formatting.

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


    # =====================================================
    # STATISTICS
    # =====================================================

    original_words = len(
        final_text.split()
    )

    summary_words = len(
        summary.split()
    )

    if original_words > 0:

        reduction = (
            1 -
            (
                summary_words /
                original_words
            )
        ) * 100

    else:

        reduction = 0


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


    # =====================================================
    # DOWNLOAD SUMMARY
    # =====================================================

    st.markdown("---")

    st.download_button(
        label="Download Summary",
        data=summary,
        file_name="smart_summary.txt",
        mime="text/plain",
        use_container_width=True
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
        NLP & Transformer Based Application
    </div>
    """,
    unsafe_allow_html=True
)
