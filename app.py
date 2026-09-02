import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import PyPDF2
import io
import re


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Smart Text & Document Summarizer",
    page_icon="📝",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }

    .summary-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #f8f9fa;
        line-height: 1.7;
    }

    .info-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #eef4ff;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


@st.cache_resource
def load_model():
    """
    Load the tokenizer and summarization model.
    The model is loaded only once and cached by Streamlit.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    return tokenizer, model, device


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):
    """
    Clean unnecessary spaces and blank lines.
    """

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------

def extract_pdf_text(uploaded_file):
    """
    Extract text from an uploaded PDF file.
    """

    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return clean_text(text)

    except Exception as error:
        st.error(f"Could not read the PDF file: {error}")

        return ""


# ---------------------------------------------------------
# TXT TEXT EXTRACTION
# ---------------------------------------------------------

def extract_txt_text(uploaded_file):
    """
    Extract text from an uploaded TXT file.
    """

    try:
        content = uploaded_file.read()

        text = content.decode("utf-8")

        return clean_text(text)

    except UnicodeDecodeError:

        try:
            uploaded_file.seek(0)

            text = uploaded_file.read().decode(
                "latin-1"
            )

            return clean_text(text)

        except Exception as error:
            st.error(
                f"Could not read the TXT file: {error}"
            )

            return ""

    except Exception as error:
        st.error(
            f"Could not read the TXT file: {error}"
        )

        return ""


# ---------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------

def split_text(text, max_words=350):
    """
    Divide long text into smaller chunks.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(
            words[i:i + max_words]
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------
# SUMMARIZATION
# ---------------------------------------------------------

def summarize_chunk(
    text,
    tokenizer,
    model,
    device,
    min_length=30,
    max_length=120
):
    """
    Summarize one text chunk.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    input_ids = inputs["input_ids"].to(device)

    attention_mask = inputs["attention_mask"].to(device)

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

    return summary


def generate_summary(text, summary_type):
    """
    Generate a short or detailed summary.
    """

    tokenizer, model, device = load_model()

    chunks = split_text(text)

    summaries = []

    progress_bar = st.progress(0)

    total_chunks = len(chunks)

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

        summaries.append(summary)

        progress_bar.progress(
            (index + 1) / total_chunks
        )

    progress_bar.empty()

    final_summary = " ".join(summaries)

    return final_summary


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("Summarizer Settings")

    summary_type = st.radio(
        "Choose Summary Type",
        ["Short", "Detailed"]
    )

    st.markdown("---")

    st.subheader("Supported Inputs")

    st.write(
        "• Pasted text\n"
        "• PDF documents\n"
        "• TXT files"
    )

    st.markdown("---")

    st.info(
        "The application uses a transformer-based "
        "summarization model."
    )


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

st.subheader("Enter Your Text")

text_input = st.text_area(
    "Paste your article, notes, report or any other text here:",
    height=250,
    placeholder="Paste your text here..."
)


st.subheader("Or Upload a Document")

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"]
)


# ---------------------------------------------------------
# GET INPUT TEXT
# ---------------------------------------------------------

document_text = ""


if uploaded_file is not None:

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):

        document_text = extract_pdf_text(
            uploaded_file
        )

    elif file_name.endswith(".txt"):

        document_text = extract_txt_text(
            uploaded_file
        )


# ---------------------------------------------------------
# DISPLAY EXTRACTED TEXT INFORMATION
# ---------------------------------------------------------

if document_text:

    st.success(
        f"Successfully extracted text from "
        f"{uploaded_file.name}"
    )

    word_count = len(
        document_text.split()
    )

    st.info(
        f"Extracted approximately {word_count} words."
    )


# ---------------------------------------------------------
# SUMMARIZE BUTTON
# ---------------------------------------------------------

st.markdown("---")

if st.button(
    "Generate Summary",
    type="primary",
    use_container_width=True
):

    # Give priority to uploaded document
    if document_text:

        final_text = document_text

    elif text_input.strip():

        final_text = clean_text(
            text_input
        )

    else:

        st.warning(
            "Please enter some text or upload "
            "a PDF/TXT file."
        )

        st.stop()


    # Check minimum text length
    if len(final_text.split()) < 30:

        st.warning(
            "Please provide at least 30 words "
            "for meaningful summarization."
        )

        st.stop()


    # Limit extremely large inputs
    words = final_text.split()

    if len(words) > 5000:

        st.warning(
            "The document is very large. "
            "Only the first 5000 words will be processed."
        )

        final_text = " ".join(
            words[:5000]
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

            st.success(
                "Summary generated successfully!"
            )

            # -------------------------------------------------
            # SUMMARY OUTPUT
            # -------------------------------------------------

            st.subheader(
                f"{summary_type} Summary"
            )

            st.markdown(
                f"""
                <div class="summary-box">
                {summary}
                </div>
                """,
                unsafe_allow_html=True
            )

            # -------------------------------------------------
            # STATISTICS
            # -------------------------------------------------

            original_words = len(
                final_text.split()
            )

            summary_words = len(
                summary.split()
            )

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

                if original_words > 0:

                    reduction = (
                        1 -
                        summary_words /
                        original_words
                    ) * 100

                    st.metric(
                        "Reduction",
                        f"{reduction:.1f}%"
                    )

            # -------------------------------------------------
            # DOWNLOAD SUMMARY
            # -------------------------------------------------

            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True
            )

        except Exception as error:

            st.error(
                "An error occurred while generating "
                "the summary."
            )

            st.exception(error)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        Smart Text & Document Summarizer |
        NLP & Transformer Based Application
    </div>
    """,
    unsafe_allow_html=True
)
