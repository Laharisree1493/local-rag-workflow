import pymupdf4llm
import re
import os
from llama_cpp import Llama

def extract_structured_sections(pdf_path):
    print("--- Step 1: Extracting layout-aware Markdown with PyMuPDF4LLM ---")
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    # Split document by headers (#, ##, ###)
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
    print("--- Step 2: Matching query to document structures ---")
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

def answer_question(query, context_title, context_text, model_path):
    print("--- Step 3: Generating localized response via Llama.cpp ---")
    # Low context limit for light processing
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
    
    prompt = f"""You are a helpful assistant answering questions strictly based on the context provided below.
If the answer cannot be found in the context, say "I cannot find the information in this section."

[Document Section: {context_title}]
{context_text}

Question: {query}
Answer:"""

    response = llm(
        prompt,
        max_tokens=256,
        temperature=0.1,
        stop=["\n\n", "Question:"]
    )
    return response['choices'][0]['text'].strip()

if __name__ == "__main__":
    # CONFIGURATION
    PDF_FILE = "sample.pdf"         # <-- Drop any PDF here and rename it to sample.pdf
    MODEL_FILE = "model.gguf"       # <-- Put your downloaded GGUF file here and rename it to model.gguf
    
    if not os.path.exists(PDF_FILE) or not os.path.exists(MODEL_FILE):
        print("\n[!] Setup required: Please make sure 'sample.pdf' and 'model.gguf' are in this folder.")
    else:
        # Run workflow
        sections = extract_structured_sections(PDF_FILE)
        print(f"Indexed {len(sections)} sections successfully.\n")
        
        query = input("Ask a question about your document: ")
        
        matched_title, matched_context = retrieve_relevant_section(query, sections)
        print(f"Targeting Section: [{matched_title}]\n")
        
        final_answer = answer_question(query, matched_title, matched_context, MODEL_FILE)
        print("\n================ ANSWER ================")
        print(final_answer)
        print("========================================")

