import os
import streamlit as st
import chromadb
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.prompts import PromptTemplate

from ingest import ingest_documents

load_dotenv()

# ==========================================
# UI Configuration & Styling
# ==========================================
st.set_page_config(page_title="Claims Intelligence Copilot", page_icon="🏦", layout="wide")

st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
    }
    .streamlit-expanderContent {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }
    
    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
    }
    
    /* Confidence Badges */
    .badge-high {
        background-color: #059669;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #d97706;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-review {
        background-color: #dc2626;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Initialize Global Settings
# ==========================================
@st.cache_resource
def setup_llama_index(_groq_api_key):
    # Set global embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Set global LLM
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=_groq_api_key)

@st.cache_resource
def get_query_engine():
    db = chromadb.PersistentClient(path=".chroma_db")
    chroma_collection = db.get_or_create_collection("policies", metadata={"hnsw:space": "cosine"})
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=Settings.embed_model,
    )
    
    # Custom QA Prompt for Adjusters
    qa_prompt_tmpl_str = (
        "You are an expert Claims Intelligence Copilot assisting insurance adjusters.\n"
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and no prior knowledge, answer the query.\n"
        "Analyze coverage, exclusions, and claim eligibility strictly based on the provided text.\n"
        "If the answer is not contained in the context, explicitly state 'I cannot determine this from the provided documents.'\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        text_qa_template=qa_prompt_tmpl
    )
    return query_engine

# ==========================================
# Sidebar: File Upload & Config
# ==========================================
with st.sidebar:
    st.title("⚙️ Configuration")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter Groq API Key:", type="password")
    else:
        st.success("Groq API Key detected in environment.")
        
    st.markdown("---")
    st.subheader("📄 Policy Documents")
    uploaded_files = st.file_uploader("Upload Policy PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process & Index Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Saving and embedding documents..."):
                os.makedirs("data", exist_ok=True)
                for file in uploaded_files:
                    with open(os.path.join("data", file.name), "wb") as f:
                        f.write(file.getbuffer())
                
                success = ingest_documents()
                if success:
                    st.success("Documents successfully ingested!")
                    # Clear query engine cache to load new data
                    get_query_engine.clear()
                else:
                    st.error("Failed to ingest documents.")

# ==========================================
# Main Chat Interface
# ==========================================
st.title("🏦 Claims Intelligence Copilot")
st.markdown("Ask natural language questions about coverage, exclusions, and claim eligibility based on the uploaded policy documents.")

if not api_key:
    st.info("Please provide a Groq API Key in the sidebar to start chatting.")
    st.stop()

# Initialize models
setup_llama_index(api_key)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "confidence" in msg and msg["confidence"]:
            conf_class = "badge-high" if msg["confidence"] == "HIGH" else "badge-medium" if msg["confidence"] == "MEDIUM" else "badge-review"
            st.markdown(f'<span class="{conf_class}">Confidence: {msg["confidence"]}</span>', unsafe_allow_html=True)
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Source Citations"):
                for source in msg["sources"]:
                    st.markdown(f"- **File:** {source['file']} (Page {source['page']})")
                    st.markdown(f"  > *\"{source['text']}...\"*")

# Chat input
if prompt := st.chat_input("Ask about coverage, e.g., 'Does this policy cover water damage?'"):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing policies..."):
            try:
                engine = get_query_engine()
                response = engine.query(prompt)
                
                # Analyze sources and confidence
                source_nodes = response.source_nodes
                
                sources = []
                max_score = 0.0
                
                for node in source_nodes:
                    score = node.score if node.score is not None else 0.0
                    max_score = max(max_score, score)
                    
                    # Extract metadata
                    metadata = node.metadata
                    file_name = metadata.get("file_name", "Unknown Document")
                    page_label = metadata.get("page_label", "Unknown Page")
                    text_snippet = node.text[:150].replace("\n", " ")
                    
                    sources.append({
                        "file": file_name,
                        "page": page_label,
                        "text": text_snippet,
                        "score": score
                    })
                
                # Determine confidence based on cosine similarity
                # Cosine similarity: closer to 1.0 is better.
                if max_score > 0.75:
                    confidence = "HIGH"
                    conf_class = "badge-high"
                elif max_score > 0.5:
                    confidence = "MEDIUM"
                    conf_class = "badge-medium"
                else:
                    confidence = "REVIEW REQUIRED"
                    conf_class = "badge-review"
                
                if len(sources) == 0:
                    confidence = "REVIEW REQUIRED"
                    conf_class = "badge-review"
                
                # Display response
                st.markdown(response.response)
                st.markdown(f'<span class="{conf_class}">Confidence: {confidence}</span>', unsafe_allow_html=True)
                
                # Display sources
                if sources:
                    with st.expander("📚 Source Citations"):
                        for source in sources:
                            st.markdown(f"- **File:** {source['file']} (Page {source['page']})")
                            st.markdown(f"  > *\"{source['text']}...\"*")
                            
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.response,
                    "confidence": confidence,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
