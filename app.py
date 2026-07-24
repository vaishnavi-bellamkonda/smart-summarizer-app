import streamlit as st
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------------------------------------------
# Page Configuration & UI Title
# -------------------------------------------------------------------
st.set_page_config(page_title="Smart Text & Document Summarizer", page_icon="📝", layout="centered")

st.title("📝 Smart Text & Document Summarizer")
st.markdown("Summarize long articles, PDFs, or meeting notes in seconds.")

# Direct model loading (bypasses pipeline task string issues completely)
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Helper function to extract text from PDF
def extract_pdf_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text

# -------------------------------------------------------------------
# Main Content Area: Input Selection
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["📄 Paste Text", "📂 Upload File"])
document_text = ""

with tab1:
    document_text = st.text_area("Paste your long-form text here:", height=250)

with tab2:
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            document_text = extract_pdf_text(uploaded_file)
        elif uploaded_file.type == "text/plain":
            document_text = str(uploaded_file.read().decode("utf-8"))
        
        st.success(f"File uploaded successfully! ({len(document_text.split())} words found)")

# -------------------------------------------------------------------
# Summarization Logic
# -------------------------------------------------------------------
if st.button("✨ Generate Summary", type="primary"):
    if not document_text.strip():
       # Calculate dynamic min/max length based on document size
                input_length = inputs["input_ids"].shape[1]
                max_len = min(150, max(40, int(input_length * 0.6)))
                min_len = min(20, max(10, int(input_length * 0.2)))

                # Generate high-quality summary
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=max_len,
                    min_length=min_len,
                    no_repeat_ngram_size=3,  # Prevents repetitive phrases
                    num_beams=4,
                    early_stopping=True
                )
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
                
                # Decode output back into text
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                st.divider()
                st.subheader("📌 Generated Summary")
                st.markdown(summary)
                
                st.download_button(
                    label="📥 Download Summary (.txt)",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")