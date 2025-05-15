import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import os
import textstat
from rouge_score import rouge_scorer
from bs4 import BeautifulSoup
import pypdf
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Financial Report Summarizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    /* Overall styling */
    .main {
        background-color: #F8F9FA;
    }
    
    /* Custom container */
    .custom-container {
        background-color: white;
        padding: 20px 30px;
        border-radius: 10px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
    }
    
    /* Headers */
    .main-title {
        color: #1E3A8A;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 10px;
        padding-bottom: 10px;
        border-bottom: 2px solid #E5E7EB;
    }
    
    .section-title {
        color: #1E3A8A;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    
    /* Status containers */
    .success-container {
        background-color: #ECFDF5;
        color: #065F46;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #10B981;
        margin-bottom: 20px;
    }
    
    .info-container {
        background-color: #EFF6FF;
        color: #1E40AF;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 20px;
    }
    
    .preview-container {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 15px;
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* Summary container */
    .summary-container {
        background-color: #FFFFFF;
        border-left: 4px solid #1E3A8A;
        border-radius: 8px;
        padding: 20px;
        margin-top: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Metrics */
    .metrics-container {
        background-color: #F9FAFB;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1E3A8A;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        margin-top: 5px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1E3A8A;
        color: white;
        border: none;
        padding: 10px 25px;
        font-weight: 500;
        border-radius: 6px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1E40AF;
        box-shadow: 0 4px 10px rgba(30, 58, 138, 0.3);
    }
    
    /* Upload area */
    .css-1kyxreq {
        border-radius: 10px !important;
        border: 2px dashed #CBD5E1 !important;
        padding: 30px !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #F1F5F9;
    }
    
    /* Remove fullscreen button from st.text_area */
    .css-1b32pqr {
        display: none;
    }
    
    /* Custom hr */
    .custom-hr {
        margin-top: 20px;
        margin-bottom: 20px;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(200, 200, 200, 0), rgba(200, 200, 200, 0.75), rgba(200, 200, 200, 0));
    }
</style>
""", unsafe_allow_html=True)

# Title with custom styling
st.markdown('<h1 class="main-title">Financial 10-K Report Summarizer</h1>',
            unsafe_allow_html=True)

# Description container
st.markdown("""
<div class="custom-container">
    <p>This tool analyzes and summarizes the Management's Discussion and Analysis (MD&A) section of financial 10-K reports. 
    Upload your report in PDF, HTML, or TXT format to generate a concise, structured summary of key financial insights.</p>
</div>
""", unsafe_allow_html=True)

# MDA Keywords for extraction
MDA_KEYWORDS = [
    "Management's Discussion and Analysis",
    "Management Discussion and Analysis",
    "MD&A",
    "Item 7.",
    "ITEM 7."
]

# Text cleaning helper function


def clean_text(text):
    """Clean up common text issues like escaped newlines"""
    # Replace escaped newlines with actual newlines
    text = text.replace('\\n', '\n')
    # Replace escaped tabs with spaces
    text = text.replace('\\t', '    ')
    # Replace escaped quotes
    text = text.replace('\\"', '"')
    # Replace escaped apostrophes
    text = text.replace("\\'", "'")
    # Clean up excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text

# Helper functions


def extract_text_from_pdf(file_content):
    """Extract text from PDF content"""
    try:
        pdf_file = BytesIO(file_content)
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        return clean_text(text)  # Apply cleaning here
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


def extract_text_from_html(file_content):
    """Extract text from HTML content"""
    try:
        content = file_content.decode('utf-8')
        soup = BeautifulSoup(content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip()
                  for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return clean_text(text)  # Apply cleaning here
    except Exception as e:
        st.error(f"Error extracting text from HTML: {e}")
        return ""


def extract_mda_section(text):
    """Extract the MD&A section from the input text"""
    # Find start of MD&A section
    start_pos = -1
    matching_keyword = None

    for keyword in MDA_KEYWORDS:
        match = re.search(r'(' + re.escape(keyword) +
                          r'[^\n]*)', text, re.IGNORECASE)
        if match:
            start_pos = match.start()
            matching_keyword = match.group(0)
            break

    if start_pos == -1:
        st.warning("MD&A section not found. Using the entire document.")
        return clean_text(text), None  # Clean the text

    # Extract a reasonable chunk starting from MD&A header
    end_pos = min(start_pos + 150000, len(text))  # Roughly 20,000 words

    # Try to find the next section (Item 8 typically follows Item 7)
    next_item_match = re.search(
        r'Item\s+8\.', text[start_pos:end_pos], re.IGNORECASE)
    if next_item_match:
        end_pos = start_pos + next_item_match.start()

    # Extract the text and clean it
    mda_text = clean_text(text[start_pos:end_pos])

    return mda_text, matching_keyword

# Load model (cached)


@st.cache_resource
def load_model(model_path):
    try:
        # Try to load the model locally first
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        st.warning(f"Could not load local model: {e}")
        st.info("Falling back to pre-trained BART model...")
        # Fallback to pre-trained model
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        return model, tokenizer


def generate_summary(text, model, tokenizer, chunk_size=900):
    """Generate summary for long MD&A text using chunking."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(tokenizer.encode(current_chunk + sentence)) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    all_summaries = []
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Processing chunk {i+1}/{len(chunks)}..."):
            inputs = tokenizer(chunk, return_tensors="pt",
                               max_length=1024, truncation=True)
            outputs = model.generate(
                inputs["input_ids"], max_length=256, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            all_summaries.append(summary.strip())

    return "\n\n".join(all_summaries)


def post_process_summary(summary, company_name=""):
    """Clean up and structure the summary"""
    # Remove repeated phrases
    summary = re.sub(r'\b(\w+)\s+\1\b', r'\1', summary)

    # Structure the summary if it's not already structured
    if "Financial Performance:" not in summary and "Business Operations:" not in summary:
        sentences = summary.split('. ')
        financial_parts = []
        operational_parts = []
        outlook_parts = []
        other_parts = []

        financial_terms = ['revenue', 'sales', 'profit',
                           'income', 'earnings', 'margin', 'billion', 'million']
        operational_terms = ['segment', 'business',
                             'market', 'product', 'service', 'operations']
        outlook_terms = ['future', 'expect', 'growth', 'forecast', 'outlook']

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if any(term in sentence.lower() for term in financial_terms):
                financial_parts.append(sentence)
            elif any(term in sentence.lower() for term in operational_terms):
                operational_parts.append(sentence)
            elif any(term in sentence.lower() for term in outlook_terms):
                outlook_parts.append(sentence)
            else:
                other_parts.append(sentence)

        # Rebuild the summary
        structured_summary = ""

        if financial_parts:
            structured_summary += "<strong>Financial Performance:</strong><br>" + \
                '. '.join(financial_parts) + ".<br><br>"

        if operational_parts:
            structured_summary += "<strong>Business Operations:</strong><br>" + \
                '. '.join(operational_parts) + ".<br><br>"

        if outlook_parts:
            structured_summary += "<strong>Outlook:</strong><br>" + \
                '. '.join(outlook_parts) + ".<br><br>"

        if other_parts:
            structured_summary += "<strong>Additional Information:</strong><br>" + \
                '. '.join(other_parts) + "."

        summary = structured_summary

    # Remove excessive bullet points (a problem in some models)
    summary = re.sub(r'(•\s*){3,}', '', summary)

    return summary


def calculate_metrics(original_text, summary):
    """Calculate evaluation metrics for the summary"""
    metrics = {}

    # Readability score
    metrics["readability"] = {
        "original_flesch_kincaid_grade": textstat.flesch_kincaid_grade(original_text),
        "summary_flesch_kincaid_grade": textstat.flesch_kincaid_grade(summary),
        "original_flesch_reading_ease": textstat.flesch_reading_ease(original_text),
        "summary_flesch_reading_ease": textstat.flesch_reading_ease(summary)
    }

    # Simple sentence splitting using regex
    def split_into_sentences(text):
        sentences = re.split(r'[.!?][\s\n]', text)
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]

    # Length statistics
    original_sentences = split_into_sentences(original_text)
    summary_sentences = split_into_sentences(summary)

    metrics["length"] = {
        "original_char_count": len(original_text),
        "summary_char_count": len(summary),
        "original_word_count": len(original_text.split()),
        "summary_word_count": len(summary.split()),
        "original_sentence_count": len(original_sentences),
        "summary_sentence_count": len(summary_sentences),
        "compression_ratio": len(original_text.split()) / max(1, len(summary.split()))
    }

    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(original_text, summary)
    metrics["rouge"] = {
        "rouge1": {
            "precision": rouge_scores['rouge1'].precision,
            "recall": rouge_scores['rouge1'].recall,
            "fmeasure": rouge_scores['rouge1'].fmeasure
        },
        "rouge2": {
            "precision": rouge_scores['rouge2'].precision,
            "recall": rouge_scores['rouge2'].recall,
            "fmeasure": rouge_scores['rouge2'].fmeasure
        },
        "rougeL": {
            "precision": rouge_scores['rougeL'].precision,
            "recall": rouge_scores['rougeL'].recall,
            "fmeasure": rouge_scores['rougeL'].fmeasure
        }
    }

    return metrics


# Sidebar configuration
with st.sidebar:
    st.markdown("## Options")

    # Model selection
    st.markdown("### Choose a model:")
    model_option = st.radio(
        "",
        ["Fine-tuned BART", "Pre-trained BART"],
        label_visibility="collapsed"
    )

    if model_option == "Fine-tuned BART":
        model_path = "data/financial-bart-finetuned"  # Path to your extracted model
    else:
        model_path = "facebook/bart-large-cnn"

    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)

    # Metrics option
    st.markdown("### Analysis Options:")
    metrics_option = st.checkbox("Calculate metrics", value=True)

    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)

    # About section
    st.markdown("### About")
    st.markdown("""
    This tool uses a fine-tuned BART model to summarize 
    financial reports. It extracts the MD&A section and 
    generates a structured summary of key financial insights.
    
    Created as part of a Master's project in Data Science.
    """)

# Main app functionality
# Create sample data folder if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# File upload section - with custom container
st.markdown('<h2 class="section-title">1. Upload or Select a 10-K Report</h2>',
            unsafe_allow_html=True)

st.markdown('<div class="custom-container">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a file", type=["pdf", "html", "txt", "htm"])

# Sample file option
# Get list of available MD&A files in the data directory
mda_files = [f for f in os.listdir("data") if f.endswith("_mda.txt")]
sample_options = ["None"] + mda_files

st.markdown("Or select a sample report:")
sample_option = st.selectbox("", sample_options, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

full_text = None

# Process the selected file
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    file_content = uploaded_file.read()

    if file_extension == "pdf":
        full_text = extract_text_from_pdf(file_content)
    elif file_extension in ["html", "htm"]:
        full_text = extract_text_from_html(file_content)
    else:  # txt file
        full_text = clean_text(file_content.decode(
            "utf-8"))  # Clean text here too

    st.markdown(
        f'<div class="success-container">File uploaded successfully: {len(full_text)} characters</div>', unsafe_allow_html=True)

elif sample_option != "None":
    # Use a sample file
    sample_file = os.path.join("data", sample_option)

    try:
        with open(sample_file, "r", encoding="utf-8") as f:
            full_text = clean_text(f.read())  # Clean the text here
        st.markdown(
            f'<div class="success-container">Using sample file: {sample_option}</div>', unsafe_allow_html=True)
    except UnicodeDecodeError:
        # Try alternative encoding if UTF-8 fails
        try:
            with open(sample_file, "r", encoding="latin-1") as f:
                full_text = clean_text(f.read())  # Clean the text here
            st.markdown(
                f'<div class="success-container">Using sample file: {sample_option}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")
    except Exception as e:
        st.error(f"Sample file not found: {sample_file}")

# Process the text if we have it
if full_text:
    # Extract MD&A section
    st.markdown('<h2 class="section-title">2. Extract MD&A Section</h2>',
                unsafe_allow_html=True)
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)

    mda_text, matching_keyword = extract_mda_section(full_text)

    if matching_keyword:
        st.markdown(
            f'<div class="success-container">Found MD&A section starting with: "{matching_keyword}"</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="info-container">Extracted {len(mda_text)} characters from the MD&A section</div>', unsafe_allow_html=True)

    with st.expander("Preview MD&A Text"):
        preview_length = min(5000, len(mda_text))
        preview_text = mda_text[:preview_length] + \
            "..." if len(mda_text) > preview_length else mda_text
        # Double check for any remaining escape sequences
        preview_text = clean_text(preview_text)
        st.markdown('<div class="preview-container">', unsafe_allow_html=True)
        st.text_area("", preview_text, height=200,
                     label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Generate summary
    st.markdown('<h2 class="section-title">3. Generate Summary</h2>',
                unsafe_allow_html=True)
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)

    generate_button = st.button("Generate Summary", use_container_width=True)

    if generate_button:
        # Load model
        with st.spinner("Loading model..."):
            model, tokenizer = load_model(model_path)

        with st.spinner(" Analyzing financial data..."):
            # Generate summary
            summary = generate_summary(mda_text, model, tokenizer)

            # Post-process summary
            processed_summary = post_process_summary(summary)

            # Display results
            st.markdown('<h2 class="section-title">4. Results</h2>',
                        unsafe_allow_html=True)
            st.markdown('<div class="summary-container">',
                        unsafe_allow_html=True)
            st.markdown(processed_summary, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Calculate metrics if requested
            if metrics_option:
                with st.spinner("📊 Calculating metrics..."):
                    metrics = calculate_metrics(
                        mda_text, re.sub('<.*?>', '', processed_summary))

                st.markdown('<div class="metrics-container">',
                            unsafe_allow_html=True)
                st.markdown(
                    '<h3 style="color: #1E3A8A; margin-bottom: 20px;">Summary Metrics</h3>', unsafe_allow_html=True)

                # Display metrics in a visually appealing way
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown('<div class="metric-card">',
                                unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-value">{metrics["length"]["compression_ratio"]:.1f}x</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="metric-label">Compression Ratio</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="metric-card">',
                                unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-value">{metrics["readability"]["summary_flesch_reading_ease"]:.1f}</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="metric-label">Reading Ease Score</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown('<div class="metric-card">',
                                unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-value">{metrics["rouge"]["rouge1"]["fmeasure"]*100:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="metric-label">ROUGE-1 F1 Score</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Add more detailed metrics in expandable section
                with st.expander("View Detailed Metrics"):
                    # Word counts
                    st.markdown("#### Document Statistics")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            f"**Original Word Count:** {metrics['length']['original_word_count']:,}")
                        st.markdown(
                            f"**Original Sentence Count:** {metrics['length']['original_sentence_count']:,}")
                        st.markdown(
                            f"**Original Flesch-Kincaid Grade:** {metrics['readability']['original_flesch_kincaid_grade']:.2f}")

                    with col2:
                        st.markdown(
                            f"**Summary Word Count:** {metrics['length']['summary_word_count']:,}")
                        st.markdown(
                            f"**Summary Sentence Count:** {metrics['length']['summary_sentence_count']:,}")
                        st.markdown(
                            f"**Summary Flesch-Kincaid Grade:** {metrics['readability']['summary_flesch_kincaid_grade']:.2f}")

                    st.markdown("#### ROUGE Scores")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**ROUGE-1**")
                        st.markdown(
                            f"Precision: {metrics['rouge']['rouge1']['precision']:.4f}")
                        st.markdown(
                            f"Recall: {metrics['rouge']['rouge1']['recall']:.4f}")
                        st.markdown(
                            f"F1: {metrics['rouge']['rouge1']['fmeasure']:.4f}")

                    with col2:
                        st.markdown("**ROUGE-2**")
                        st.markdown(
                            f"Precision: {metrics['rouge']['rouge2']['precision']:.4f}")
                        st.markdown(
                            f"Recall: {metrics['rouge']['rouge2']['recall']:.4f}")
                        st.markdown(
                            f"F1: {metrics['rouge']['rouge2']['fmeasure']:.4f}")

                    with col3:
                        st.markdown("**ROUGE-L**")
                        st.markdown(
                            f"Precision: {metrics['rouge']['rougeL']['precision']:.4f}")
                        st.markdown(
                            f"Recall: {metrics['rouge']['rougeL']['recall']:.4f}")
                        st.markdown(
                            f"F1: {metrics['rouge']['rougeL']['fmeasure']:.4f}")

                st.markdown('</div>', unsafe_allow_html=True)

            # Download option - with custom styling
            plain_summary = re.sub('<.*?>', '', processed_summary)
            st.download_button(
                "Download Summary",
                plain_summary,
                "summary.txt",
                "text/plain",
                use_container_width=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; color: #6B7280; font-size: 0.8rem;">
    Financial 10-K Report Summarizer • Master's Project in Data Science • 2025
</div>
""", unsafe_allow_html=True)
