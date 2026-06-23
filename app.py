import streamlit as st
import pymupdf4llm
import re
from huggingface_hub import InferenceClient

# Initialize Hugging Face Serverless Client (Runs completely free in the cloud)
# This replaces the heavy local llama-cpp engine that crashes Streamlit's free tier due to RAM limits
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")

def extract_structured_sections(pdf_path):
    md_text = pymupdf4llm.to_markdown(pdf_path)
    header_regex = re.compile(r'(^#{1,4}\s+.*$)', re.MULTILINE)
    parts = header_regex.split(md_text)
    
    sections = {}
    current_header = "Introduction / Overview"
    sections[current_header] = ""
    
    for part in parts:
        if header_regex.match(part):
            current_header = part.strip().lstrip('#').strip()
            sections[current_header] = ""
        else:
            sections[current_header] += part
    return sections

def retrieve_relevant_section(query, sections):
    query_words = set(re.findall(r'\w+', query.lower()))
    best_match_header = None
    max_overlap = 0
    
    for header in sections.keys():
        header_words = set(re.findall(r'\w+', header.lower()))
        overlap = len(query_words.intersection(header_words))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match_header = header
            
    if not best_match_header or max_overlap == 0:
        best_match_header = list(sections.keys())[0]
    return best_match_header, sections[best_match_header]

# --- STREAMLIT UI CODE ---
st.set_page_config(page_title="Mozilla Lightweight RAG", layout="centered")
st.title("📄 Mozilla.ai Lightweight Document Q&A")
st.write("Upload a structured PDF document to query it using a Roaming RAG architecture.")

# File Uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the file temporarily to read it
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Parse sections
    if "sections" not in st.session_state:
        with st.spinner("Processing document structures with PyMuPDF4LLM..."):
            st.session_state.sections = extract_structured_sections("temp_doc.pdf")
        st.success(f"Indexed {len(st.session_state.sections)} sections successfully!")

    # Search Bar Query
    user_query = st.text_input("Ask a question about your document:")
    
    if user_query:
        # Match section
        matched_title, matched_context = retrieve_relevant_section(user_query, st.session_state.sections)
        
        st.info(f"🎯 **Targeted Section:** {matched_title}")
        
        # Call API for answer generation
        with st.spinner("Generating answer..."):
            prompt = f"""You are a helpful assistant answering questions strictly based on the context provided below.
If the answer cannot be found in the context, say "I cannot find the information in this section."

[Document Section: {matched_title}]
{matched_context}

Question: {user_query}
Answer:"""

            try:
                response = client.text_generation(prompt, max_new_tokens=256, temperature=0.1)
                st.subheader("💡 Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"API Error: {e}. Try asking again.")
